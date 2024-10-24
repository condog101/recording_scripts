import open3d as o3d
import numpy as np


def rough_register_via_correspondences(source_cloud, target_points):
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    # estimate rough transformation using correspondences

    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    corr = np.array([[i, i] for i in range(len(target_points))])
    trans_init = p2p.compute_transformation(
        source_cloud, target, o3d.utility.Vector2iVector(corr))

    threshold = 0.002  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_cloud, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return reg_p2p.transformation
    # point-to-point ICP for refinement
    # print("Perform point-to-point ICP refinement")
    # threshold = 0.03  # 3cm distance threshold
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # draw_registration_result(source, target, reg_p2p.transformation)
    # draw_registration_result(source, target, trans_init)
