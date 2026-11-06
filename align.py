import open3d as o3d
import numpy as np


def _preprocess_for_global_registration(pcd, voxel_size):
    """下采样 + 法线 + FPFH 特征，用于粗配准。"""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return pcd_down, fpfh


def _global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """使用基于特征的 RANSAC 获取包含旋转的初始位姿。"""
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def _refine_icp(source, target, init_transformation, voxel_size):
    """在粗配准基础上进行 ICP 精配准。"""
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=50)
    )
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=50)
    )

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance=voxel_size * 0.6,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    return result


def process_and_normalize(path1, path2, out1, out2):
    print("正在加载并归一化点云...")
    pcd1 = o3d.io.read_point_cloud(path1)
    pcd2 = o3d.io.read_point_cloud(path2)

    # 1. 居中
    center1 = pcd1.get_center()
    center2 = pcd2.get_center()
    pcd1.translate(-center1)
    pcd2.translate(-center2)

    # 2. 归一化缩放 (关键步骤！)
    # 将模型缩放到最大跨度为 1.0
    extent = np.max(pcd1.get_max_bound() - pcd1.get_min_bound())
    scale_factor = 20.0 / extent
    pcd1.scale(scale_factor, center=(0, 0, 0))
    pcd2.scale(scale_factor, center=(0, 0, 0))
    print(f"已将模型缩放 {scale_factor:.6f} 倍 (原始跨度: {extent:.2f} -> 目标跨度: 1.0)")

    # 3. 下采样 (可选，为了流畅度)
    # pcd1 = pcd1.uniform_down_sample(5) 
    # pcd2 = pcd2.uniform_down_sample(5)

    # 4. 先做全局粗配准（解决大旋转），再做 ICP 精配准
    voxel_size = 0.25
    pcd1_down, pcd1_fpfh = _preprocess_for_global_registration(pcd1, voxel_size)
    pcd2_down, pcd2_fpfh = _preprocess_for_global_registration(pcd2, voxel_size)

    print("开始全局粗配准（RANSAC + FPFH）...")
    coarse = _global_registration(pcd2_down, pcd1_down, pcd2_fpfh, pcd1_fpfh, voxel_size)
    print(
        f"粗配准完成: fitness={coarse.fitness:.4f}, rmse={coarse.inlier_rmse:.4f}"
    )

    print("开始 ICP 精配准...")
    fine = _refine_icp(pcd2, pcd1, coarse.transformation, voxel_size)
    print(f"精配准完成: fitness={fine.fitness:.4f}, rmse={fine.inlier_rmse:.4f}")

    pcd2.transform(fine.transformation)

    # 5. 保存
    o3d.io.write_point_cloud(out1, pcd1, write_ascii=False)
    o3d.io.write_point_cloud(out2, pcd2, write_ascii=False)
    print("处理完成！现在的模型尺寸非常适合网页显示。")

if __name__ == "__main__":
    process_and_normalize(
        "resources/3d_vis/vggt/scannet.ply", 
        "resources/3d_vis/vggt+ours/scannet.ply",
        "resources/3d_vis/vggt/scannet.ply",
        "resources/3d_vis/vggt+ours/scannet.ply"
    )