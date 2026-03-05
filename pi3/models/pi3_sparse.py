import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.sparse_attention import SparseFlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin
from einops import rearrange
from torch.utils.checkpoint import checkpoint


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False

def forward_block_rearrangement(x, H, W, patch_size, sparse_block_size):
    h = H // patch_size
    w = W // patch_size
    
    # Reshape to 2D grid
    x_grid = rearrange(x, 'b (h w) d -> b h w d', h=h, w=w)
    
    # Reorder into block-major sequence
    x_blocked = rearrange(
        x_grid,
        'b (nh ph) (nw pw) d -> b (nh nw ph pw) d',
        nh = h // sparse_block_size,
        nw = w // sparse_block_size,
        ph = sparse_block_size,
        pw = sparse_block_size
    )
    return x_blocked

def backward_block_rearrangement(
    x: torch.Tensor,
    H: int,
    W: int,
    patch_size: int,
    sparse_block_size: int
) -> torch.Tensor:
    # Height and width of the feature map in terms of patches
    h_patches = H // patch_size
    w_patches = W // patch_size

    nh = h_patches // sparse_block_size
    nw = w_patches // sparse_block_size
    
    ph = sparse_block_size
    pw = sparse_block_size

    x_grid = rearrange(
        x, 'b (nh nw ph pw) d -> b (nh ph) (nw pw) d',
        nh=nh, nw=nw, ph=ph, pw=pw
    )

    x_original_order = rearrange(x_grid, 'b h w d -> b (h w) d')

    return x_original_order

class Pi3_Sparse(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        self.sparse_block_size = 4
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024

        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.local_decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth // 2)])
        
        self.global_decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=SparseFlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth // 2)])

        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token (removed in speed3r)
        # ----------------------
        # num_register_tokens = 5
        # self.patch_start_idx = num_register_tokens
        # self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        # nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        # freeze_all_params([self.encoder, self.point_decoder, self.conf_decoder, self.camera_decoder])
        freeze_all_params([self.encoder])



    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        mid_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        # rearange vit to make patch-wise continious
        pos = forward_block_rearrangement(pos, H, W, 14, self.sparse_block_size)
        hidden = forward_block_rearrangement(hidden, H, W, 14, self.sparse_block_size)
       
        for i in range(len(self.global_decoder)):
            local_blk = self.local_decoder[i]
            global_blk = self.global_decoder[i]
            
            pos = pos.reshape(B*N, hw, -1)
            hidden = hidden.reshape(B*N, hw, -1)
            if self.training:
                hidden = checkpoint(local_blk, hidden, pos, use_reentrant=False)
            else:
                hidden = local_blk(hidden, xpos=pos)
            if i == len(self.global_decoder) - 1:
                final_output.append(hidden.reshape(B*N, hw, -1))
            if i == len(self.global_decoder) // 2 - 1:
                mid_output.append(hidden.reshape(B*N, hw, -1))


            pos = pos.reshape(B, N, hw, -1).reshape(B*N, hw, -1)
            hidden = hidden.reshape(B, N, hw, -1).reshape(B*N, hw, -1)

            pos = pos.reshape(B, N*hw, -1)
            hidden = hidden.reshape(B, N*hw, -1)
            if self.training:
                hidden = checkpoint(global_blk, hidden, pos, use_reentrant=False)
            else:
                hidden = global_blk(hidden, xpos=pos)

            pos = pos.reshape(B, N, hw, -1).reshape(B*N, hw, -1)
            hidden = hidden.reshape(B, N, hw, -1).reshape(B*N, hw, -1)

            if i == len(self.global_decoder) - 1:
                final_output.append(hidden.reshape(B*N, hw, -1))
            if i == len(self.global_decoder) // 2 - 1:
                mid_output.append(hidden.reshape(B*N, hw, -1))

        pos = backward_block_rearrangement(pos, H, W, self.patch_size, self.sparse_block_size)
        final_output[0] = backward_block_rearrangement(final_output[0], H, W, self.patch_size, self.sparse_block_size)
        final_output[1] = backward_block_rearrangement(final_output[1], H, W, self.patch_size, self.sparse_block_size)
        mid_output[0] = backward_block_rearrangement(mid_output[0], H, W, self.patch_size, self.sparse_block_size)
        mid_output[1] = backward_block_rearrangement(mid_output[1], H, W, self.patch_size, self.sparse_block_size)

        final_output_cat = torch.cat([final_output[0], final_output[1]], dim=-1)
        pos_reshaped = pos.reshape(B*N, hw, -1)
        mid_output_cat = torch.cat([mid_output[0].reshape(B, N, hw, -1), mid_output[1].reshape(B, N, hw, -1)], dim=-1)

        return final_output_cat, pos_reshaped, mid_output_cat
    
    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, pos, mid_hidden = self.decode(hidden, N, H, W)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden], (H, W)).reshape(B, N, H, W, -1)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden, patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            mid_hidden=mid_hidden
        )


if __name__ == '__main__':
    model = Pi3_Sparse().to('cuda:0')
    B, N, H, W = 1, 8, 224, 448
    print(B, N, H , W)
    for i in range(1):
        x = torch.rand([B, N, 3, H, W], device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
        with torch.amp.autocast(device_type='cuda', enabled=True):
        # model.decode(x, N, H, W)
            out = model(x)
        
        loss = (torch.rand_like(out['points']) - out['points']).mean()
        print(loss)
        loss.backward()
        import pdb; pdb.set_trace()

        