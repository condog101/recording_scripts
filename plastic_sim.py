import open3d as o3d

import open3d as o3d
from argparse import ArgumentParser
import numpy as np
from markup_sequence import pick_mesh_points, pick_points
from armature_utils import get_joint_positions
from visualization_utils import create_arrow
from joints import create_armature_objects, Spine
from collections import defaultdict
from registration import rough_register_via_correspondences, source_icp_transform
from utils import get_closest_indices, find_points_within_radius
from surg_shape_simulator import flip_normals
from scipy.spatial import cKDTree
import copy


def color_clusters(mesh, clusters, cluster_n_triangles):

    n_clusters = len(cluster_n_triangles)

    # Generate distinct colors for each cluster
    cluster_colors = np.random.random((n_clusters, 3))

    # Map colors to vertices based on triangle clusters
    vertex_colors = np.zeros((np.asarray(mesh.vertices).shape[0], 3))
    triangles = np.asarray(mesh.triangles)

    for i in range(len(triangles)):
        cluster_id = clusters[i]
        vertex_colors[triangles[i]] = cluster_colors[cluster_id]

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


def cull_small_clusters(mesh, clusters, cluster_n_triangles):
    small_clusters = np.argwhere(
        np.array(cluster_n_triangles) < 1000).reshape(-1)
    invalid_clusters = np.argwhere(
        np.isin(clusters, small_clusters)).reshape(-1)
    return invalid_clusters


def get_spine_obj(spine_mesh, partition_map, spine_spline_indices, scene_points, sampled_cloud, spine_points, pcd_spine, new_partition_map, splits=2):
    cleaned_spine_points = np.asarray(spine_mesh.vertices)[
        spine_spline_indices]

    reverse_partition_map = {}
    for key, vals in partition_map.items():
        for val in vals:
            reverse_partition_map[val] = key

    links = get_joint_positions(
        partition_map, spine_mesh, splits=splits, spine_points=spine_points)
    root_node = create_armature_objects(
        links, new_partition_map, cleaned_spine_points)
    root_node.set_parents()

    new_root_node = create_armature_objects(
        links, partition_map, cleaned_spine_points)
    new_root_node.set_parents()

    return Spine(root_node, np.asarray(
        pcd_spine.points), cleaned_spine_points, scene_points, sampled_cloud, alpha=1.0, beta=0, threshold=10, normal_cloud=sampled_cloud, tip_threshold=0.7), Spine(new_root_node, np.asarray(
            spine_mesh.vertices), cleaned_spine_points, scene_points, sampled_cloud, alpha=0.9, beta=0.1, threshold=0.4, normal_cloud=sampled_cloud, tip_threshold=0.4)


def subsample_mesh(mesh, clusters, points):
    sample_clusters = []
    for ind, triangle in enumerate(np.asarray(mesh.triangles)):
        for vert_index in triangle:
            if vert_index in points:
                sample_clusters.append(clusters[ind])

    return np.argwhere(
        ~np.isin(clusters, np.array(sample_clusters))).reshape(-1)


def get_joints_with_axes(spine_obj):
    all_joints = spine_obj.get_all_joints()
    axes = [joint.get_axes() for joint in all_joints]

    spheres = []
    arrows = []
    for ind, joint in enumerate(all_joints):

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
        spheres.append(sphere.translate(joint.position))
        vectors = axes[ind]

        for ind2, vector in enumerate(vectors):
            color = np.zeros(3)
            color[ind2] = 1
            arrows.append(create_arrow(joint.position,
                          vector, scale=20, color=color))
    return spheres, arrows


def get_rough_transform(source_points, scene_points):
    source_whole_rough = o3d.geometry.PointCloud()
    source_whole_rough.points = o3d.utility.Vector3dVector(source_points)

    source_scene_rough = o3d.geometry.PointCloud()
    source_scene_rough.points = o3d.utility.Vector3dVector(
        scene_points)

    return rough_register_via_correspondences(
        source_whole_rough, source_scene_rough)


def initial_mesh_processing(mesh):
    mesh_vertices = np.asarray(mesh.vertices)*1000
    mesh_vertices[:, 2] = mesh_vertices[:, 2] * -1.0
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)

    # compute principal components, length wise will be x axis, most similar to z unit vector will be z
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(triangle_clusters)

    invalid_triangles = cull_small_clusters(
        mesh, clusters, cluster_n_triangles)
    mesh.remove_triangles_by_index(invalid_triangles)
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(triangle_clusters)
    color_clusters(mesh, clusters, cluster_n_triangles)
    return clusters


def subsample_and_color(mesh, clusters, picked_process_points):
    invalid_triangles = subsample_mesh(mesh, clusters, picked_process_points)
    mesh.remove_triangles_by_index(invalid_triangles)
    mesh.remove_unreferenced_vertices()
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(triangle_clusters)
    color_clusters(mesh, clusters, cluster_n_triangles)
    return clusters


def find_nearest_neighbors_kdtree(source_points, target_points):
    """
    Find nearest neighbors using KD-tree for better performance with large datasets.
    """
    tree = cKDTree(target_points)
    distances, indices = tree.query(source_points, k=1)
    return indices, distances


def pick_all_points(mesh, pcd, clusters, repick):
    plastic_curve_points_path = f"plastic_curve_points.npy"
    plastic_correspondences_path = f"plastic_correspondences.npy"
    if repick:
        print("Select relevent vertebrae (using points on spine process)")
        picked_process_points = pick_mesh_points(mesh)
        picked_vertices = np.asarray(mesh.vertices)[picked_process_points]
        with open(plastic_curve_points_path, 'wb+') as f:
            # source, destination
            np.save(f, np.array(picked_vertices))
        # reduce down to only relevent vertebrae
        clusters = subsample_and_color(mesh, clusters, picked_process_points)

        picked_process_points = get_closest_indices(
            picked_vertices, np.asarray(mesh.vertices))
        print("Select correspondence points")
        picked_mesh_points_indices = pick_mesh_points(mesh)
        picked_cloud_points_indices = pick_points(pcd)
        picked_mesh_correspondence_vertices = np.asarray(mesh.vertices)[
            picked_mesh_points_indices]
        picked_cloud_points = np.asarray(pcd.points)[
            picked_cloud_points_indices]

        if len(picked_mesh_points_indices) != len(picked_cloud_points_indices):
            raise ValueError(f"Correspondences are of different length!")
        combined = np.array(
            [picked_mesh_correspondence_vertices, picked_cloud_points])
        with open(plastic_curve_points_path, 'wb+') as f:
            np.save(f, np.array(picked_vertices))

        with open(plastic_correspondences_path, 'wb+') as f:
            np.save(f, combined)

    else:
        with open(plastic_curve_points_path, 'rb') as f:
            picked_vertices = np.load(f)

        picked_process_points = get_closest_indices(
            picked_vertices, np.asarray(mesh.vertices))

        clusters = subsample_and_color(mesh, clusters, picked_process_points)

        picked_process_points = get_closest_indices(
            picked_vertices, np.asarray(mesh.vertices))

        with open(plastic_correspondences_path, 'rb') as f:
            combined = np.load(f)
            picked_mesh_correspondence_vertices = combined[0]
            picked_cloud_points = combined[1]

    return picked_process_points, picked_vertices, picked_mesh_correspondence_vertices, picked_cloud_points, clusters


def main(path, mesh_path, repick):

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    triangles = np.asarray(mesh.triangles)
    triangles = np.flip(triangles, axis=1)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    pcd = o3d.io.read_point_cloud(path)
    clusters = initial_mesh_processing(mesh)

    # use this picking to define which processes we're choosing
    picked_process_points, picked_vertices, picked_mesh_correspondence_vertices, picked_cloud_points, clusters = pick_all_points(
        mesh, pcd, clusters, repick)

    pcd_spine = o3d.geometry.PointCloud()
    pcd_spine.points = mesh.vertices
    pcd_spine.normals = mesh.vertex_normals

    threshold = 0.3

    rough_transform = get_rough_transform(picked_mesh_correspondence_vertices,
                                          picked_cloud_points)

    icp_transform = source_icp_transform(
        pcd_spine, pcd, rough_transform, threshold=threshold, plane=False)
    mesh.transform(icp_transform)
    o3d.visualization.draw_geometries([mesh, pcd])
    pcd_spine = mesh.sample_points_uniformly(
        number_of_points=20000)
    pcd_spine = mesh.sample_points_poisson_disk(
        number_of_points=3000, pcl=pcd_spine)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                             std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(25)
    normals = np.asarray(pcd.normals)
    flipped_normals = flip_normals(normals)
    pcd.normals = o3d.utility.Vector3dVector(flipped_normals)
    matching_cloud, matching_indices = find_points_within_radius(
        pcd, pcd_spine, radius=4)
    quantile = np.quantile(np.asarray(matching_cloud.points)[:, 2], 0.9)
    rel_indices = np.argwhere(np.asarray(
        matching_cloud.points)[:, 2] < quantile).reshape(-1)
    matching_cloud = matching_cloud.select_by_index(rel_indices)

    # take only top 3rd of this

    # need to extract triangles from
    partition_map = defaultdict(set)
    triangles = np.asarray(mesh.triangles)
    [partition_map[key].add(t) for ind, key in enumerate(
        clusters) for t in triangles[ind]]
    # map from vert to spine cluster
    reverse_partition_map = {t: key for ind, key in enumerate(
        clusters) for t in triangles[ind]}
    partition_map = {key: list(value) for key, value in partition_map.items()}

    pcd_spine = mesh.sample_points_uniformly(
        number_of_points=30000)
    pcd_spine = mesh.sample_points_poisson_disk(
        number_of_points=9000, pcl=pcd_spine)

    nearest_neighbours, _ = find_nearest_neighbors_kdtree(
        np.asarray(pcd_spine.points), np.asarray(mesh.vertices))
    new_partition_map = defaultdict(list)
    for ind, point in enumerate(np.asarray(pcd_spine.points)):
        correspondence = reverse_partition_map[nearest_neighbours[ind]]
        new_partition_map[correspondence].append(ind)

    spine_points = np.asarray(mesh.vertices)[picked_process_points]

    spine_obj, spine_obj_to_deform = get_spine_obj(mesh, partition_map, picked_process_points, np.asarray(
        matching_cloud.points), matching_cloud, spine_points, pcd_spine, new_partition_map,  splits=1)
    fparams, _ = spine_obj.run_optimization()
    joint_params = fparams[:-6]
    global_params = fparams[-6:]
    joint_params, global_params = spine_obj.convert_degree_params_to_radians(
        joint_params, global_params)
    spine_obj_to_deform.apply_global_parameters(global_params)
    spine_obj_to_deform.apply_global_transform()
    spine_obj_to_deform.apply_joint_parameters(joint_params)
    spine_obj_to_deform.apply_joint_angles()
    new_mesh = copy.deepcopy(mesh)
    new_mesh.vertices = o3d.utility.Vector3dVector(
        spine_obj_to_deform.vertices)
    spheres, arrow = get_joints_with_axes(spine_obj_to_deform)
    o3d.visualization.draw_geometries([new_mesh, pcd])
    # L4,L3,L2,L1
    # saved fparam
#     fparams = [ 0.99931738, -0.00841635, -0.03497295,  0.00841635,  0.99931206,  0.00902543,
#   0.03482073, -0.00902543,  0.99931738, -0.00841635,  0.03497295, -0.00841635,
#   0.99931206, -0.00902543,  0.03482073,  0.00902543,  2.426605,    1.31965527,
#  -2.8169578 ]
    # T12, T11, T10, T9
    fparams = np.array([-0.048600106773572266, -0.018691688254530762, 0.005026687885406139, -0.007472760605150189, 0.06125811554466146, -0.023759268154223523, 0.07, -0.0424630274776413,
                       0.017453591154585682, -0.007302153158121103, 0.0011836297053235655, 0.006253535159680209, -0.08642878672776477, 0.23486927197167684, 0.0066370604067586565])

    # L2, L1, T12
    fparams = np.array([-0.003701871571250792, 0.1060793732633063, 0.004317122450332389, 0.25, -
                        0.08534906738116482, 0.2170913567681204, 0.0007975652988245773, 0.0, 0.000983811906848329, -2.5, -1, 2])

    fparams = np.array([33, 2, -39, 40, -16, 6, -3, 0, 0, 0, -2, -1])


if __name__ == "__main__":
    parser = ArgumentParser(description="surgery simulation")
    parser.add_argument(
        "FILE", type=str, help="Path to pcd file")
    args = parser.parse_args()
    filename: str = args.FILE
    # /home/connorscomputer/Documents/fragment_of_fragment.pcd
    # /home/connorscomputer/Documents/fragment_of_fragment_side.pcd
    phantom_mesh_path = "/home/connorscomputer/phantom_mesh.ply"
    main(filename, phantom_mesh_path, False)
