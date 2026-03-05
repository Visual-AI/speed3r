from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange
from pi3.models.sparse_attn.flash_attention_topk import flash_attn_topk_func
from pi3.models.sparse_attn.topk_blocksparse_attention import topk_blocksparse_attention

import torch

class AttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.rope = rope

    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SparseFlashAttentionRope(AttentionRope):

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,    
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None,
        block_size: int = 4,
    ) -> None:
        super().__init__(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop, qk_norm, norm_layer, rope)
        # gate function
        gate_linear_layer = torch.nn.Linear(dim, self.num_heads * 2, bias=True)
        nn.init.zeros_(gate_linear_layer.weight)
        bias_init = torch.tensor([-1.0, 1.0]).repeat(self.num_heads)
        gate_linear_layer.bias.data.copy_(bias_init)


        self.gate = torch.nn.Sequential(
            gate_linear_layer,
            torch.nn.Sigmoid(),
        )
        self.block_size = block_size


    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        gate = self.gate(x)
        gate = rearrange(gate, "b n (h g) -> b h n g", g=2)

        # Compress sequence dimension
        qkv_compress = rearrange(qkv, 'b n (qkv h d) -> (b qkv h) d n', qkv=3, h=self.num_heads)
        qkv_compress = F.avg_pool1d(qkv_compress, kernel_size=self.block_size**2, stride=self.block_size**2)
        qkv_compress = rearrange(qkv_compress, '(b qkv h) d n -> b h n qkv d', b=B, qkv=3, h=self.num_heads)

        # Split q, k, v correctly
        q_compress, k_compress, v_compress = qkv_compress.unbind(dim=-2)  # Split along the qkv dimension
        q_compress, k_compress = self.q_norm(q_compress).to(v_compress.dtype), self.k_norm(k_compress).to(v_compress.dtype)

        # rope for block
        if xpos is not None:
            xpos_blocked = xpos.view(B,  N // self.block_size**2, self.block_size**2, 2)
            xpos_compress = xpos_blocked.float().mean(dim=2).int()
            q_compress = self.rope(q_compress, xpos_compress)
            k_compress = self.rope(k_compress, xpos_compress)

        compressed_attn_output, idx = flash_attn_topk_func(q_compress, k_compress, v_compress, 32)

        # rope for token
        qkv_full = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv_full[:,:,i] for i in range(3)]
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if xpos is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
        sparse_attn_output = topk_blocksparse_attention(q, k, v, idx, block_size=self.block_size**2)

        sparse_attn_output_blocked = rearrange(sparse_attn_output, 'b h (n_c blk) d -> b h n_c blk d', blk=self.block_size**2)
        gate_blocked = rearrange(gate, 'b h (n_c blk) g -> b h n_c blk g', blk=self.block_size**2)
        compressed_attn_output_broadcastable = rearrange(compressed_attn_output, 'b h n_c d -> b h n_c 1 d')

        attn_output_blocked = (
            gate_blocked[..., 0:1] * compressed_attn_output_broadcastable
            + gate_blocked[..., 1:2] * sparse_attn_output_blocked
        )
        
        attn_output = rearrange(attn_output_blocked, 'b h n_c blk d -> b (n_c blk) (h d)')


        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        return attn_output
    

if __name__ == '__main__':
    attn = SparseFlashAttentionRope(dim=1024)
    attn.to('cuda:0')
    B, N, C= 1, 336//14 * 504//14 * 16, 1024
    x = torch.rand([B, N, C], dtype=torch.bfloat16).to('cuda:0')
    # model.decode(x, N, H, W)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = attn(x)