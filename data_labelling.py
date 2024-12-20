import open3d as o3d
import open3d.visualization as vis
from argparse import ArgumentParser
import numpy as np
from markup_sequence import pick_mesh_points, pick_points, mark_relevant_frames
from armature_utils import get_joint_positions
from visualization_utils import create_arrow
# from joints import create_armature_objects, Spine
from joints_tail_root import create_armature_objects, Spine
from collections import defaultdict
from registration import rough_register_via_correspondences, source_icp_transform
from utils import get_closest_indices, find_points_within_radius, partition, sample_points_inside_mesh, filter_points_in_bbox
from surg_shape_simulator import flip_normals, flip_normals_z
from manual_param_visualizer import InteractiveMeshVisualizer
from scipy.spatial import cKDTree
import pickle as pkl
import copy
import cv2
from markup_sequence import get_baseline_frame, subsample_pcd_with_bounding_box, label_sequence
from pyk4a import PyK4APlayback
import trimesh
import json

OUTPUT_MESH_FOLDER = "/home/connorscomputer/Documents/CT_DATA/CT_NIFTIS/NIFTI_SOLIDS_TRANSFORMED/"
CONFIG_FOLDER = "/home/connorscomputer/Documents/CT_DATA/CONFIGS_BASIN/"
RESULTS = "/home/connorscomputer/Documents/CT_DATA/RESULTS/"


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


def get_spine_obj(spine_mesh, partition_map, spine_spline_indices, scene_points, sampled_cloud, spine_points, pcd_spine, new_partition_map, config, splits=2):
    cleaned_spine_points = np.asarray(spine_mesh.vertices)[
        spine_spline_indices]
    mean_spine = np.mean(cleaned_spine_points, axis=0)
    centroid = np.mean(np.asarray(spine_mesh.vertices), axis=0)
    upvector = (mean_spine - centroid)/np.linalg.norm((mean_spine - centroid))
    reverse_partition_map = {}
    for key, vals in partition_map.items():
        for val in vals:
            reverse_partition_map[val] = key

    links = get_joint_positions(
        partition_map, spine_mesh, splits=splits, spine_points=spine_points)
    root_node = create_armature_objects(
        links, new_partition_map, cleaned_spine_points, upvector, np.asarray(pcd_spine.points))
    root_node.set_parents()

    new_root_node = create_armature_objects(
        links, partition_map, cleaned_spine_points, upvector, np.asarray(spine_mesh.vertices))
    new_root_node.set_parents()

    return Spine(root_node, np.asarray(
        pcd_spine.points), cleaned_spine_points, scene_points, sampled_cloud, upvector, config, normal_cloud=sampled_cloud), Spine(new_root_node, np.asarray(
            spine_mesh.vertices), cleaned_spine_points, scene_points, sampled_cloud, upvector, config, normal_cloud=sampled_cloud), Spine(new_root_node, np.asarray(
                spine_mesh.vertices), cleaned_spine_points, scene_points, sampled_cloud, upvector, config, normal_cloud=sampled_cloud)


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


def pick_all_points(mesh, pcd, clusters, short_file_path, repick):
    curve_points_path = f"curve_points_{short_file_path}.npy"
    correspondences_path = f"correspondences_{short_file_path}.npy"
    if repick:
        print("Select vertebrae points on spine process")
        picked_process_points = pick_mesh_points(mesh)
        picked_vertices = np.asarray(mesh.vertices)[picked_process_points]
        with open(curve_points_path, 'wb+') as f:
            # source, destination
            np.save(f, np.array(picked_vertices))

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
        with open(curve_points_path, 'wb+') as f:
            np.save(f, np.array(picked_vertices))

        with open(correspondences_path, 'wb+') as f:
            np.save(f, combined)

    else:
        with open(curve_points_path, 'rb') as f:
            picked_vertices = np.load(f)

        picked_process_points = get_closest_indices(
            picked_vertices, np.asarray(mesh.vertices))

        picked_process_points = get_closest_indices(
            picked_vertices, np.asarray(mesh.vertices))

        with open(correspondences_path, 'rb') as f:
            combined = np.load(f)
            picked_mesh_correspondence_vertices = combined[0]
            picked_cloud_points = combined[1]

    return picked_process_points, picked_vertices, picked_mesh_correspondence_vertices, picked_cloud_points


def visualize_with_joints(mesh, arrows, spheres):
    mat_box = vis.rendering.MaterialRecord()
    # mat_box.shader = 'defaultLitTransparency'
    mat_box.shader = 'defaultLitSSR'
    mat_box.base_color = [0.467, 0.467, 0.467, 0.2]
    mat_box.base_roughness = 0.0
    mat_box.base_reflectance = 0.0
    mat_box.base_clearcoat = 1.0
    mat_box.thickness = 1.0
    mat_box.transmission = 1.0
    mat_box.absorption_distance = 0.5
    mat_box.absorption_color = [0.5, 0.5, 0.5]

    mat_spheres = []
    for ind, sphere in enumerate(spheres):
        mat_sphere = vis.rendering.MaterialRecord()
        mat_sphere.shader = 'defaultLit'

        mat_sphere.base_color = np.array([0.5, 0.5, 0.5, 1]).reshape(4, 1)
        mat_spheres.append(
            {'name': f"sphere_{ind}", 'geometry': sphere, 'material': mat_box})

    mat_arrows = []
    for ind, arrow in enumerate(arrows):
        mat_arrow = vis.rendering.MaterialRecord()
        mat_arrow.shader = 'defaultLit'
        colors = list(
            np.asarray(arrow.vertex_colors)[0])
        colors.append(1.0)
        colors = np.array(colors).reshape(4, 1)
        mat_arrow.base_color = colors
        mat_arrows.append(
            {'name': f"arrow_{ind}", 'geometry': arrow, 'material': mat_arrow})

    mesh_attr = {'name': 'box', 'geometry': mesh, 'material': mat_box}

    geoms = [mesh_attr] + mat_arrows + mat_spheres
    vis.draw(geoms)


def disconnect_clusters(mesh):
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    to_remove = []
    for ind, triangle in enumerate(triangles):
        color = colors[triangle]
        if not np.all(color == color[0], axis=1).all():
            to_remove.append(ind)
    mesh.remove_triangles_by_index(to_remove)
    mesh.remove_unreferenced_vertices()
    return


def fill_holes_mesh(unprocessed_mesh: trimesh.Trimesh):
    unprocessed_mesh.fix_normals()

    # Remove duplicate and degenerate faces
    unprocessed_mesh.remove_duplicate_faces()
    unprocessed_mesh.remove_degenerate_faces()

    # Final cleanup
    unprocessed_mesh.merge_vertices()
    unprocessed_mesh.fill_holes()
    vertices = np.array(unprocessed_mesh.vertices)
    faces = np.array(unprocessed_mesh.faces)

    # Create open3d triangle mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Compute normals for proper rendering
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def main(videopath, mesh_path, repick, baseline_frame=0):

    file_short = videopath.split('/')[-1].split('.')[0]

    mesh = o3d.io.read_triangle_mesh(mesh_path)

    unprocessed_mesh = trimesh.load(mesh_path, force='mesh')
    unprocessed_mesh = fill_holes_mesh(unprocessed_mesh)

    img, pcd, calibration, transformed_depth = get_baseline_frame(
        videopath, baseline_frame)
    coordinates, baseline_frame = mark_relevant_frames([img], 0)
    # rmove this!
    # coordinates[0] = (270, 149, (1, 1, 1))
    # coordinates[1] = (719, 423, (1, 1, 1))

    x_tl, y_tl, color = coordinates[0]
    x_br, y_br, _ = coordinates[1]

    cv2.rectangle(img, (x_tl, y_tl), (x_br, y_br), (255, 255, 0), 2)
    cv2.imshow('Image with Bounding Box', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pcd = subsample_pcd_with_bounding_box(
        pcd, coordinates, calibration, transformed_depth)
    # subsample baseline frame to get initial mesh
    disconnect_clusters(mesh)

    o3d.io.write_triangle_mesh(OUTPUT_MESH_FOLDER +
                               f"mesh_untransformed_undeformed_{file_short}.ply", mesh)
    transformed_mesh_undeformed = copy.deepcopy(mesh)
    partition_map = partition(mesh)
    rgb_num_map = {key: ind for ind, key in enumerate(partition_map)}
    # replace keys in partition maps with
    for key, value in rgb_num_map.items():
        partition_map[value] = partition_map.pop(key)

    # use this picking to define which processes we're choosing
    # first pick processes from bottom to top, then correspondences
    picked_process_points, picked_vertices, picked_mesh_correspondence_vertices, picked_cloud_points = pick_all_points(
        mesh, pcd, partition_map, file_short, repick)

    pcd_spine = o3d.geometry.PointCloud()
    pcd_spine.points = mesh.vertices
    pcd_spine.normals = mesh.vertex_normals

    threshold = 0.3

    rough_transform = get_rough_transform(picked_mesh_correspondence_vertices,
                                          picked_cloud_points)

    icp_transform = source_icp_transform(
        pcd_spine, pcd, rough_transform, threshold=threshold, plane=False)
    reverse_icp_transform = np.linalg.inv(icp_transform)
    mesh.transform(icp_transform)
    unprocessed_mesh.transform(icp_transform)
    o3d.visualization.draw_geometries([mesh, pcd])

    inside_sample = sample_points_inside_mesh(o3d.t.geometry.TriangleMesh.from_legacy(
        unprocessed_mesh), 5000, 1, unprocessed_mesh.get_axis_aligned_bounding_box())
    pcd_spine = mesh.sample_points_uniformly(
        number_of_points=20000)
    pcd_spine = mesh.sample_points_poisson_disk(
        number_of_points=3000, pcl=pcd_spine)

    pod = o3d.geometry.PointCloud()
    pod.points = o3d.utility.Vector3dVector(np.concatenate(
        [np.asarray(pcd_spine.points), inside_sample]))

    inside_pcd = o3d.geometry.PointCloud()
    inside_pcd.points = o3d.utility.Vector3dVector(inside_sample)
    matching_cloud, matching_indices = find_points_within_radius(
        pcd, pod, radius=0.4)

    bbox2 = matching_cloud.get_axis_aligned_bounding_box()
    matching_cloud = filter_points_in_bbox(pcd, bbox2)

    matching_cloud.estimate_normals()
    try:
        matching_cloud.orient_normals_consistent_tangent_plane(25)
    except RuntimeError:
        print("Couldn't orient normals")
    normals = np.asarray(matching_cloud.normals)
    flipped_normals = flip_normals(np.asarray(matching_cloud.points), normals)
    flipped_normals = flip_normals_z(flipped_normals)
    matching_cloud.normals = o3d.utility.Vector3dVector(flipped_normals)

    quantile = np.quantile(np.asarray(matching_cloud.points)[:, 2], 0.9)
    rel_indices = np.argwhere(np.asarray(
        matching_cloud.points)[:, 2] < quantile).reshape(-1)
    matching_cloud = matching_cloud.select_by_index(rel_indices)

    reverse_partition_map = {}
    for key, vals in partition_map.items():
        for val in vals:
            reverse_partition_map[val] = key

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

    with open(f"{CONFIG_FOLDER}deform_config_{file_short}.json", 'r') as f:
        config = json.load(f)

    spine_obj, spine_obj_to_deform, spine_obj_to_deform_untransformed = get_spine_obj(mesh, partition_map, picked_process_points, np.asarray(
        matching_cloud.points), matching_cloud, spine_points, pcd_spine, new_partition_map, config,  splits=1)

    # spheres, arrow = get_joints_with_axes(spine_obj)
    # visualize_with_joints(mesh, arrow, spheres)

    fparams = config['initial_fparam']

    # fparams, _ = spine_obj.run_optimization()

    new_mesh = copy.deepcopy(mesh)
    # viz = InteractiveMeshVisualizer()
    # viz.set_mesh_and_spine_obj(
    #     new_mesh, spine_obj_to_deform, fparams, pcd)
    # viz.run()
    # print("")

    spine_obj_to_deform_untransformed.apply_global_transform_world_coordinates(
        reverse_icp_transform)

    joint_params = fparams[:-6]
    global_params = fparams[-6:]
    joint_params, global_params = spine_obj.convert_degree_params_to_radians(
        joint_params, global_params)

    spine_obj_to_deform_untransformed.apply_joint_parameters(joint_params)
    spine_obj_to_deform_untransformed.apply_joint_angles()

    new_mesh_untransformed = copy.deepcopy(mesh)
    new_mesh_untransformed.vertices = o3d.utility.Vector3dVector(
        spine_obj_to_deform_untransformed.vertices)
    spine_obj_to_deform.reset_spine()
    spine_obj_to_deform.apply_global_parameters(global_params)
    spine_obj_to_deform.apply_global_transform()

    new_mesh_undeformed = copy.deepcopy(mesh)
    new_mesh_undeformed.vertices = o3d.utility.Vector3dVector(
        spine_obj_to_deform.vertices)

    o3d.io.write_triangle_mesh(OUTPUT_MESH_FOLDER +
                               f"mesh_transformed_undeformed{file_short}.ply", new_mesh_undeformed)

    all_joints = spine_obj_to_deform.get_all_joints()
    axes = np.array([[*joint.get_axes(), joint.position]
                    for joint in all_joints])

    save_axes_params(axes, fparams, file_short)

    spine_obj_to_deform.apply_joint_parameters(joint_params)
    spine_obj_to_deform.apply_joint_angles()

    # spheres, arrow = get_joints_with_axes(spine_obj_to_deform)

    new_mesh.vertices = o3d.utility.Vector3dVector(
        spine_obj_to_deform.vertices)

    source, target = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    source.points, target.points = o3d.utility.Vector3dVector(np.asarray(new_mesh_untransformed.vertices)[
        :100]), o3d.utility.Vector3dVector(np.asarray(new_mesh.vertices)[:100])

    o3d.io.write_triangle_mesh(OUTPUT_MESH_FOLDER +
                               f"mesh_transformed_{file_short}.ply", new_mesh)

    # rigid_transform = rough_register_via_correspondences(source, target)

    # we want to save mesh, fparams, joint coordinates and joint axes vectors

    o3d.visualization.draw_geometries([new_mesh, pcd])

    recorded_points = label_sequence(
        new_mesh, videopath, coordinates, transform_window_size=30, threshold=0.7)
    recorded_points_array = [np.array((x[0], x[1])) for x in recorded_points]

    file_points = f"{RESULTS}closest_points_{file_short}.pkl"
    with open(file_points, 'wb+') as f:
        # source, destination
        pkl.dump(recorded_points_array, f)


def save_axes_params(axes, fparams, file_short):

    file_fparams = f"{RESULTS}fparams_{file_short}.npy"
    file_axes = f"{RESULTS}axes_{file_short}.npy"
    with open(file_fparams, 'wb+') as f:
        # source, destination
        np.save(f, fparams)

    with open(file_axes, 'wb+') as f:
        # source, destination
        np.save(f, axes)


if __name__ == "__main__":

    videopath = "/home/connorscomputer/Documents/CT_DATA/VIDEOS/MXdLc6ez_20240911_091846.mkv"
    meshpath = "/home/connorscomputer/Documents/CT_DATA/CT_NIFTIS/NIFTI_SOLIDS/mesh_MXdLc6ez_20240911_091846.ply"
    # parser = ArgumentParser(description="Data labelling file")
    # parser.add_argument(
    #     "FILE1", type=str, help="Path to mkv file")
    # parser.add_argument(
    #     "FILE2", type=str, help="Path to mesh file")
    # args = parser.parse_args()
    # videopath: str = args.FILE1
    # meshpath = args.FILE2

    main(videopath, meshpath, False, baseline_frame=0)

[3.16789025e+00,  3.50000000e+00, -3.07109098e-03,  2.57858830e+00,
 3.22681595e+00, -1.41716013e+00, -1.26055501e+01, -2.10188174e+00,
 6.04643178e+00, 1.00000000e+00,  3.73652842e+00,  2.48284774e+00]
