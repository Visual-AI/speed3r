import sys
sys.path.append('.')

from datasets.base.base_dataset import BaseDataset
import os
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
from datasets.base.transforms import *
from datasets.base.utils import read_depth
import gzip
import json

def convert_ndc_to_pinhole(focal_length, principal_point, image_size):
    focal_length = np.array(focal_length)
    principal_point = np.array(principal_point)
    image_size_wh = np.array([image_size[1], image_size[0]])
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    fx, fy = focal_length_px[0], focal_length_px[1]
    cx, cy = principal_point_px[0], principal_point_px[1]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K

def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    R = R[None, :, :]
    T = T[None, :]
    focal = focal[None, :]
    p0 = p0[None, :]
    image_size = image_size[None, :]

    R_pytorch3d = R.copy()
    T_pytorch3d = T.copy()
    focal_pytorch3d = focal
    p0_pytorch3d = p0
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.transpose(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size[:, ::-1]

    # NDC to screen conversion.
    scale = np.min(image_size_wh, axis=1, keepdims=True) / 2.0
    scale = np.repeat(scale, 2, axis=1)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = np.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R[0], tvec[0], camera_matrix[0]

class CO3DV2Dataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        mask_bg='rand',
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'COD3DV2'
        self.data_root = data_root

        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        if not os.path.exists(f'{self.data_root}/co3dv2_{self.mode}_cache.npy'):
            self.sequences = []
            self.num_image = {}

            for seq in tqdm(os.listdir(osp.join(data_root, 'co3d_seq_downsampled_1x'))):
                try:
                    annotation_path_train = osp.join(data_root, 'co3d_anno_seq_downsampled_1x', seq + '_train.jgz')
                    annotation_path_test = osp.join(data_root, 'co3d_anno_seq_downsampled_1x', seq + '_test.jgz')
                    if self.mode == 'train':
                        with gzip.open(annotation_path_train, 'rt', encoding='utf-8') as f:
                            annotation = json.load(f)
                    else:
                        with gzip.open(annotation_path_test, 'rt', encoding='utf-8') as f:
                            annotation = json.load(f)
                except:
                    continue

                for sub_seq in annotation.keys():
                    self.num_image[(seq, sub_seq)] = len(annotation[sub_seq])
                    self.sequences.append((seq, sub_seq))

            np.save(f'{self.data_root}/co3dv2_{self.mode}_cache', dict(sequences=self.sequences, num_image=self.num_image))

        else:
            npy = np.load(f'{self.data_root}/co3dv2_{self.mode}_cache.npy', allow_pickle=True).item()
            self.sequences = npy['sequences']
            self.num_image = npy['num_image']

        # self.sequences = sorted(self.sequences)

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences of {self.dataset_label} dataset:', self.sequences)

        print(f'[{self.dataset_label}] Found {len(self.sequences)} unique videos in {data_root}', flush=True)

    def __len__(self):
        return len(self.sequences)
                    
    def _get_views(self, index, resolution, rng):
        scene = tuple(self.sequences[index])

        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))

        annotation_path_train = osp.join(self.data_root, 'co3d_anno_seq_downsampled_1x', scene[0] + '_train.jgz')
        annotation_path_test = osp.join(self.data_root, 'co3d_anno_seq_downsampled_1x', scene[0] + '_test.jgz')

        with gzip.open(annotation_path_train, 'rt', encoding='utf-8') as f:
            annotation_train = json.load(f)

        with gzip.open(annotation_path_test, 'rt', encoding='utf-8') as f:
            annotation_test = json.load(f)

        annotation = {**annotation_train, **annotation_test}[scene[1]]

        num_img = len(annotation)
        should_replace = num_img < self.frame_num
        idxs = rng.choice(num_img, self.frame_num, replace=should_replace)
        
        self.this_views_info = dict(
            scene=scene,
            idxs=idxs,
        )

        views = []
        for idx in idxs:
            anno = annotation[idx]
            filepath = anno['filepath']

            impath = osp.join(self.data_root, 'co3d_seq_downsampled_1x',filepath)
            depthpath = osp.join(self.data_root, 'co3d_seq_downsampled_1x', filepath.replace('images', 'depths')+'.geometric.png')

            # load image and depth
            rgb_image = np.array(Image.open(impath))

            # depthmap = read_depth(depthpath)
            depthmap = np.ones_like(rgb_image[:,:,0]) * 100.0
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

            # load camera params
            camera_intrinsics = np.array(anno['intri'])
            world_to_camera_pose = np.array(anno['extri'])
            bottom_row = np.array([[0, 0, 0, 1]])
            world_to_camera_pose = np.concatenate([world_to_camera_pose, bottom_row], axis=0)
            
            # camera_pose = world_to_camera_pose
            camera_pose = np.linalg.inv(world_to_camera_pose)

            if mask_bg:
                # load object mask
                maskpath = impath.replace('/images/', '/masks/').replace('.jpg', '.png')
                maskmap = np.array(Image.open(maskpath)).astype(np.float32)
                maskmap = maskmap > 0.1

                # update the depthmap with mask
                depthmap *= maskmap

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, camera_intrinsics.copy(), resolution, rng=rng, info=impath)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_label,
                label=scene[0]+'-'+scene[1],
                instance=osp.split(impath)[1],
            ))
        return views
    
