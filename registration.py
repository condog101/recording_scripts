import open3d as o3d
import numpy as np


def rough_register_via_correspondences(source_cloud, target):

    # estimate rough transformation using correspondences

    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    corr = np.array([[i, i]
                    for i in range(len(target.points))]).astype(np.float64)
    trans_init = p2p.compute_transformation(
        source_cloud, target, o3d.utility.Vector2iVector(corr))

    return trans_init  # reg_p2p.transformation


def source_icp_transform(source, target, trans_init, threshold=5, plane=True):
    if not plane:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    else:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return reg_p2p.transformation


def invert_icp_point_to_plane(source, target, trans_init, threshold=5):
    reverse_init_transform = np.linalg.inv(trans_init)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        target, source, threshold, reverse_init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    reversed_transform = reg_p2l.transformation
    return np.linalg.inv(reversed_transform)


def preprocess_point_cloud(pcd, voxel_size, estimate_normals=True):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    if estimate_normals:
        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, spine_pcd_source, surgical_pcd_target):
    print(":: Load two point clouds and disturb initial pose.")

    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(
        spine_pcd_source, voxel_size, estimate_normals=False)
    target_down, target_fpfh = preprocess_point_cloud(
        surgical_pcd_target, voxel_size)
    return spine_pcd_source, surgical_pcd_target, source_down, target_down, source_fpfh, target_fpfh


def force_below_z_threshold(spine_mesh, curve_points, offset=20):
    min_z_value_scene = np.min(curve_points[:, 2])
    min_z_value_pcd = np.min(np.asarray(spine_mesh.vertices)[:, 2])
    transform = np.eye(4)
    if min_z_value_scene > min_z_value_pcd:
        diff = min_z_value_scene - min_z_value_pcd
        transform[2, 3] = diff + offset
    return transform
