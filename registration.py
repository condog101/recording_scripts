import open3d as o3d
import numpy as np


def rough_register_via_correspondences(source_cloud, target):

    # estimate rough transformation using correspondences

    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    corr = np.array([[i, i] for i in range(len(target.points))])
    trans_init = p2p.compute_transformation(
        source_cloud, target, o3d.utility.Vector2iVector(corr))

    threshold = 0.01  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_cloud, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return trans_init  # reg_p2p.transformation


def source_icp_transform(source, target, trans_init, threshold=5):

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation


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


# voxel_size = 0.5  # means 5cm for this dataset
# source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
#     voxel_size)

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f"
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 10
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(800000, 1000))
    return result
