from .base.transforms import *

# Import necessary libraries for the fix
import torch.distributed as dist
from torch.utils.data import DataLoader

from utils.misc import get_world_size, get_rank
import hydra
from datasets.base.base_dataset import sample_resolutions, unified_collate_fn
# Import all your samplers, including the new one we will create
from datasets.base.batched_sampler import DynamicBatchSampler, DynamicDistributedSampler, SingleProcessDynamicSampler

__HIGH_QUALITY_DATASETS__ = ['BlinkVision', 'Game', 'GameNew', 'DynamicStereo', 'FlyingThings3D', 'GTA-sfm', 'Hypersim', 'MatrixCity', 'MidAir', 'Monkaa', 'PointOdyssey', 'Sintel', 'Spring', 'TarTanAir', 'Unreal4k', 'VirtualKitti', 'Habitat']
__MIDDLE_QUALITY_DATASETS__ = ['BlendedMVG', 'BlendedMVS', 'DTU', 'ETH3D', 'ScanNet', 'Scannetpp', 'Taskonomy']
__INDOOR_DATASETS__ = ['Hypersim', 'ScanNet', 'Scannetpp', 'Taskonomy', 'Arkitscene', 'Habitat']

def create_dataloader(cfg, mode):
    data_loader = DataLoader

    # pytorch dataset
    if mode == 'train':
        cfg_dataset = cfg.train_dataset
        cfg_dataloader = cfg.train_dataloader
        batch_size = cfg.train.batch_size
        num_workers = cfg.train.num_workers
    else:
        cfg_dataset = cfg.test_dataset
        cfg_dataloader = cfg.test_dataloader
        batch_size = cfg.test.batch_size
        num_workers = cfg.test.num_workers

    if isinstance(cfg_dataset, str):
        dataset = eval(cfg_dataset) 
    elif 'weights' in cfg_dataset:
        weights = cfg_dataset.weights
        if 'length' in cfg_dataset:
            dataset_length = cfg_dataset.length
            weight_sum = sum([v for k, v in weights.items()])
            new_weights = {}
            for dataset_name, weight in weights.items():
                new_weights[dataset_name] = max(int(weight / weight_sum * dataset_length), 1)
            weights = new_weights
            print(f'New weights for dataset (adjusting to dataset length {dataset_length}): {new_weights}')

        datasets_all = []

        num_resolution = cfg.train.num_resolution if 'num_resolution' in cfg.train and mode == 'train' else 1
        if mode == 'train' and 'random_reslution' in cfg.train and cfg.train.random_reslution:
            seed = 777 + 0
            resolutions = sample_resolutions(aspect_ratio_range=cfg.train.aspect_ratio_range, pixel_count_range=cfg.train.pixel_count_range, patch_size=cfg.train.patch_size, num_resolutions=num_resolution, seed=seed)
            print('Initialized resolution', resolutions)
            num_resolution = len(resolutions)
            for dataset_name, weight in weights.items():
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name], resolution=resolutions)
                dataset_i.convert_attributes()
                datasets_all.append(weight @ dataset_i)
        elif 'resolution' in cfg.train:
            resolutions = cfg.train.resolution
            print('Setting dataset resolution', resolutions)
            for dataset_name, weight in weights.items():
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name], resolution=resolutions)
                dataset_i.convert_attributes()
                datasets_all.append(weight @ dataset_i)
        else:
            for dataset_name, weight in weights.items():
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name])
                dataset_i.convert_attributes()
                datasets_all.append(weight @ dataset_i)
        dataset = datasets_all[0]
        for dataset_ in datasets_all[1:]:
            dataset += dataset_
    else:
        dataset = hydra.utils.instantiate(cfg_dataset)
        dataset.convert_attributes()

    # --- START OF CORRECTED LOGIC ---
    
    is_distributed = dist.is_available() and dist.is_initialized()
    
    image_num_range = cfg.train.image_num_range if mode == 'train' else [8, 8]
    max_img_per_gpu = cfg.train.max_img_per_gpu if 'max_img_per_gpu' in cfg.train else image_num_range[0]

    if is_distributed:
        # --- DISTRIBUTED PATH (your original code) ---
        print("INFO: Using DynamicDistributedSampler for multi-GPU training.")
        world_size = get_world_size()
        rank = get_rank()
        
        sampler = DynamicDistributedSampler(dataset, seed=cfg.train.base_seed, shuffle=cfg_dataloader.shuffle, rank=rank, drop_last=cfg_dataloader.drop_last)
        
    else:
        # --- SINGLE-PROCESS PATH (THE FIX) ---
        # Use our new non-distributed sampler that mimics the necessary behavior.
        print("INFO: Using SingleProcessDynamicSampler for single-process training.")
        world_size = 1
        rank = 0 # DynamicBatchSampler needs a rank for its RNG
        
        sampler = SingleProcessDynamicSampler(dataset, shuffle=cfg_dataloader.shuffle, seed=cfg.train.base_seed)

    # This part is now common for both paths, as `sampler` has the required interface
    print(f'Sampling frame number range from {image_num_range}')
    print(f'Max frame number per rank {max_img_per_gpu}')
    if mode == 'train' and cfg.train.iters_per_epoch > 0:
        print('Needed batch number per epoch (per rank):', (max_img_per_gpu // image_num_range[0]) * cfg.train.iters_per_epoch)
        print('Dataset length per rank:', len(dataset) // world_size)
        assert (max_img_per_gpu // image_num_range[0]) * cfg.train.iters_per_epoch < len(dataset) // world_size

    # DynamicBatchSampler works with both our new sampler and the original one
    batch_sampler = DynamicBatchSampler(
        sampler, 
        num_resolution, 
        image_num_range, 
        seed=cfg.train.base_seed,
        max_img_per_gpu=max_img_per_gpu,
        rank=rank  # Pass rank for both cases
    )

    return data_loader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=unified_collate_fn
    )
    # --- END OF CORRECTED LOGIC ---