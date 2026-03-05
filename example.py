import torch
import argparse
import time
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
import os
from pi3.models.pi3_sparse import Pi3_Sparse

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    model = Pi3_Sparse.from_pretrained("weining17/Speed3R_Pi3").to(device).eval()

    # 2. Prepare input data
    # The load_images_as_tensor function will print the loading path
    imgs = load_images_as_tensor(args.data_path, interval=args.interval, round_patch=56).to(device) # (N, 3, H, W)

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    start_time = time.perf_counter()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None]) # Add batch dimension

    # 4. process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.2
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    points = res['points'][0][masks]
    colors = imgs.permute(0, 2, 3, 1)[masks]

    max_points = 1_000_000
    num_points = points.shape[0]
    if num_points > max_points:
        indices = torch.randperm(num_points, device=points.device)[:max_points]
        points = points[indices]
        colors = colors[indices]

    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.synchronize(device)
    time_needed = time.perf_counter() - start_time

    # 5. Save points
    print(f"Saving point cloud to: {args.save_path}")
    write_ply(points.cpu(), colors, args.save_path)
    print(f"Reconstruction time needed: {time_needed:.4f} seconds")
    print("Done.")