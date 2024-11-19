from argparse import ArgumentParser
import cv2
import numpy as np
from pyk4a import PyK4APlayback, ImageFormat
import open3d as o3d
import pickle
from scipy.spatial.transform import Rotation
from typing import Optional
from utils import info, filter_z_offset_points, partition, average_quaternions, find_points_within_radius, get_new_mapping
from markup_sequence import record_pointed_spots, get_3d_coordinates, raw_get_3d_coordinates, play_single_initial_frame_mark_both
from registration import rough_register_via_correspondences, source_icp_transform
from armature_utils import get_joint_positions
from joints import create_armature_objects, Spine
import copy
from visualization_utils import create_arrow
from catmull_rom import fit_catmull_rom


def get_scene_geometry_from_capture(capture):
    capture._color = cv2.cvtColor(cv2.imdecode(
        capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
    capture._color_format = ImageFormat.COLOR_BGRA32
    points = capture.transformed_depth_point_cloud.reshape(
        (-1, 3)).astype('float64')
    colors = capture.color[..., (2, 1, 0)].reshape(
        (-1, 3))

    return colors, points


def get_3d_coords_from_stored_2d(coordinate_list, frame_index, capture):
    frame_2d_coordinates = coordinate_list[frame_index]
    frame_3d_coordinates = get_3d_coordinates(
        frame_2d_coordinates, capture.transformed_depth, capture._calibration)
    return frame_3d_coordinates


def get_min_max_x_y(coords_3d):
    min_x = 100000000
    min_y = 100000000
    max_x = -100000000
    max_y = -100000000
    for coord in coords_3d:
        x, y, _ = coord
        min_x = min(x, min_x)
        min_y = min(y, min_y)
        max_x = max(x, max_x)
        max_y = max(y, max_y)
    return (min_x, min_y, max_x, max_y)


# this is horrific and needs rewriting
def subsample_mesh(colors, points, bbox_params, calibration, depth_map, offset_y=0, offset_x=0, z_percentile=0.05):
    # try filtering bottom 20% z values
    z_vals = np.sort(points[:, 2])
    zlen = int(z_percentile * len(z_vals))
    z_cap = z_vals[:len(z_vals)-zlen][-1]

    min_x = bbox_params[0][0]
    min_y = bbox_params[0][1]
    max_x = min_x + bbox_params[1]
    max_y = min_y + bbox_params[2]
    # define coords in clockwise order tl, tr, br, bl
    point_coords = [(min_x, min_y), (max_x, min_y),
                    (max_x, max_y), (min_x, max_y)]
    coords_3d = [raw_get_3d_coordinates(x, depth_map, calibration)
                 for x in point_coords]
    filter_coords = get_min_max_x_y(coords_3d)
    mask_x = (points[:, 0] > (filter_coords[0] + offset_x)
              ) & (points[:, 0] < (filter_coords[2] - offset_x))

    mask_y = (points[:, 1] > (filter_coords[1] + offset_y)
              ) & (points[:, 1] < (filter_coords[3] - offset_y))

    mask_z = (points[:, 2] < z_cap)

    combined_mask = mask_x & mask_y & mask_z

    return colors[combined_mask], points[combined_mask], coords_3d


def get_closest_spine_indices(spine_vertex_points, spine_mesh):
    spine_vertices = np.asarray(spine_mesh.vertices)
    matching_indices = []
    print("getting closest points")
    for point in spine_vertex_points:
        norms = np.linalg.norm(spine_vertices - point, axis=1)
        print(np.min(norms))
        matching_indices.append(
            np.argmin(norms))
    return np.array(matching_indices)


def get_rough_transform(source_points, scene_points, scale_scene):
    source_whole_rough = o3d.geometry.PointCloud()
    source_whole_rough.points = o3d.utility.Vector3dVector(source_points)

    source_scene_rough = o3d.geometry.PointCloud()
    source_scene_rough.points = o3d.utility.Vector3dVector(
        scene_points * scale_scene)

    return rough_register_via_correspondences(
        source_whole_rough, source_scene_rough)


# 1/11/2024 - need to re-record points to check correspondence issue

def clean_mesh(spine_mesh, partition):
    reverse_mapping = {v: key for key, list_item in partition.items()
                       for v in list_item}
    bad_triangles = []
    for ind, triangle in enumerate(np.asarray(spine_mesh.triangles)):
        ref_vert = triangle[0]
        cluster = reverse_mapping[ref_vert]

        if not all([cluster == reverse_mapping[x] for x in triangle[1:]]):
            bad_triangles.append(ind)

    spine_mesh.remove_triangles_by_index(bad_triangles)
    spine_mesh.remove_unreferenced_vertices()
    return


def get_deformed_mesh(spine_mesh, partition_map, spine_spline_indices, curve_surg_points, scene_points, sampled_cloud, use_cache, spine_pcd):
    cleaned_spine_points = np.asarray(spine_mesh.vertices)[
        spine_spline_indices]

    spine_pcd_points = np.asarray(spine_pcd.points)
    reverse_partition_map = {}
    for key, vals in partition_map.items():
        for val in vals:
            reverse_partition_map[val] = key

    new_mapping = get_new_mapping(reverse_partition_map, np.asarray(
        spine_mesh.vertices), spine_pcd_points)
    new_mapping = {key: list(item) for key, item in new_mapping.items()}

    links = get_joint_positions(
        partition_map, spine_mesh)
    root_node = create_armature_objects(
        links, new_mapping, cleaned_spine_points)
    root_node.set_parents()

    if use_cache:
        with open(f"fparam_cache.npy", 'rb') as f:
            # source, destination
            fparams = np.load(f)
    else:
        spine_obj = Spine(root_node, spine_pcd_points,
                          curve_surg_points, scene_points, sampled_cloud)
        fparams, _ = spine_obj.run_optimization()
        with open(f"fparam_cache.npy", 'wb+') as f:
            # source, destination
            np.save(f, np.array(fparams))

    new_root_node = create_armature_objects(
        links, new_mapping, cleaned_spine_points)
    new_root_node.set_parents()
    spine_obj_original = Spine(new_root_node, np.asarray(
        spine_mesh.vertices), curve_surg_points, scene_points, sampled_cloud)
    # before we apply

    joint_params = fparams[:-7]
    global_params = fparams[-7:]
    spine_obj_original.apply_global_parameters(global_params)
    spine_obj_original.apply_global_transform()
    spine_obj_original.apply_joint_parameters(joint_params)
    spine_obj_original.apply_joint_angles()

    new_mesh = copy.deepcopy(spine_mesh)
    new_mesh.vertices = o3d.utility.Vector3dVector(
        spine_obj_original.vertices)

    # clean_mesh(new_mesh, partition_map)

    return new_mesh, spine_obj_original


def get_average_transform(transform, sliding_window, window_size=5):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    r = Rotation.from_matrix(np.array(rotation))
    quaternion = r.as_quat(scalar_first=True)

    if len(sliding_window) == 0:
        sliding_window.append((translation, quaternion))
        return transform, sliding_window
    else:

        if len(sliding_window) == window_size:
            sliding_window.pop(0)
        translations = [x[0] for x in sliding_window]
        rotations = [x[1] for x in sliding_window]
        translations.append(translation)
        avg_translation = np.mean(np.array(translations), axis=0)

        # try basic rotation averaging
        rotations.append(quaternion)

        avg_rotation = average_quaternions(np.array(rotations))
        sliding_window.append((translation, quaternion))

        rotation_matrix_back = Rotation.from_quat(
            avg_rotation, scalar_first=True).as_matrix()
        ema_matrices = np.eye(4)
        ema_matrices[:3, :3] = rotation_matrix_back
        ema_matrices[:3, 3] = avg_translation
        return ema_matrices, sliding_window


def play(playback: PyK4APlayback, offset: float, record_spots: bool, file_short: str, mesh_filepath: Optional[str] = None,
         baseline_frame: int = 0, record_mesh_points: bool = False, scale_scene=1.0, record_spline_points=False):

    if record_spots:
        record_pointed_spots(playback, file_short)

    coordinates_file = f"stored_coordinates_{file_short}.pkl"
    time_file = f"times_{file_short}.pkl"

    with open(coordinates_file, 'rb') as f:
        coordinates = pickle.load(f)

    with open(time_file, 'rb') as f:
        times = pickle.load(f)

    spine_mesh = o3d.io.read_triangle_mesh(mesh_filepath)

    if record_mesh_points:
        if spine_mesh is None:
            raise ValueError(
                f"Spine mesh filepath is wrong, cannot set points")
        print("Fitting corresponding points")
        pp_list = play_single_initial_frame_mark_both(
            spine_mesh, playback, offset, baseline_frame)

        print(pp_list)
        with open(f"meshpoints_{file_short}.npy", 'wb+') as f:
            # source, destination
            np.save(f, np.array(pp_list))
    else:
        with open(f"meshpoints_{file_short}.npy", 'rb') as f:
            # source, destination
            pp_list = np.load(f)

    if record_spline_points:
        if spine_mesh is None:
            raise ValueError(
                f"Spine mesh filepath is wrong, cannot set points")
        print("Fitting spline curve points")
        spline_list = play_single_initial_frame_mark_both(
            spine_mesh, playback, offset, baseline_frame)

        print(spline_list)
        with open(f"spline_points_{file_short}.pkl", 'wb+') as f:
            # source, destination
            pickle.dump(spline_list, f)
    else:
        with open(f"spline_points_{file_short}.pkl", 'rb') as f:
            # source, destination
            spline_list = pickle.load(f)

    # pplist, for array is source, second is target

    spine_points_spline = spline_list[0]
    surg_points_spline = spline_list[1]

    curve_surg_points = fit_catmull_rom(
        filter_z_offset_points(surg_points_spline))

    partition_map = partition(spine_mesh)

    rgb_num_map = {key: ind for ind, key in enumerate(partition_map)}
    # replace keys in partition maps with
    for key, value in rgb_num_map.items():
        partition_map[value] = partition_map.pop(key)

    source_points = pp_list[0]
    scene_points = pp_list[1]
    rough_transform = get_rough_transform(
        source_points, scene_points, scale_scene)

    spine_spline_indices = get_closest_spine_indices(
        spine_points_spline, spine_mesh)

    spine_correspondence_indices = get_closest_spine_indices(
        source_points, spine_mesh)

    geometry = o3d.geometry.PointCloud()
    whole_thing = o3d.geometry.PointCloud()

    frame_index = 0

    first = True

    # pcd for mesh from CT - this is correct scale, camera needs scaling
    pcd_spine = o3d.geometry.PointCloud()
    pcd_spine.points = spine_mesh.vertices
    pcd_spine.normals = spine_mesh.vertex_normals

    # do registration for 1st

    visualize = False
    view_static = True
    use_deformed = True

    reg_results = []

    if visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
    spine_obj = None

    rolling_correspondences = 0
    rolling_rmse = 0
    sliding_window = []
    try:
        playback.open()

        info(playback)
        if offset != 0.0:
            playback.seek(int(offset * 1000000))
        while True:
            try:
                capture = playback.get_next_capture()
                if capture.color is not None and capture.depth is not None:

                    x_c, y_c, w_c, h_c, _ = coordinates[frame_index]
                    colors, points = get_scene_geometry_from_capture(capture)
                    # x direction is up down spine
                    # y is across
                    subsample_colors, subsample_points, bbox_3d = subsample_mesh(
                        colors, points, ((x_c, y_c), w_c, h_c), capture._calibration, capture.transformed_depth, offset_y=-10, offset_x=-5, z_percentile=0.05)

                    geometry.points = o3d.utility.Vector3dVector(
                        subsample_points)
                    geometry.colors = o3d.utility.Vector3dVector(
                        (subsample_colors/255).astype('float64'))

                    whole_thing.points = o3d.utility.Vector3dVector(
                        points)
                    whole_thing.colors = o3d.utility.Vector3dVector(
                        (colors/255).astype('float64'))

                    threshold = 0.4

                    icp_transform = source_icp_transform(
                        pcd_spine, geometry, rough_transform, threshold=threshold, plane=False)

                    evaluation = o3d.pipelines.registration.evaluate_registration(
                        pcd_spine, geometry, threshold, icp_transform)

                    icp_transform, sliding_window = get_average_transform(
                        icp_transform, sliding_window)

                    reg_results.append(evaluation)

                    spine_mesh.transform(icp_transform)

                    if first:
                        if use_deformed:
                            old_mesh = copy.copy(spine_mesh)
                            subsample_colors, subsample_points, bbox_3d = subsample_mesh(
                                colors, points, ((x_c, y_c), w_c, h_c), capture._calibration, capture.transformed_depth, offset_y=-2, offset_x=9, z_percentile=0.05)

                            pcd_spine = spine_mesh.sample_points_uniformly(
                                number_of_points=30000)
                            pcd_spine = spine_mesh.sample_points_poisson_disk(
                                number_of_points=6000, pcl=pcd_spine)
                            matching_cloud, matching_indices = find_points_within_radius(
                                whole_thing, pcd_spine, radius=8)

                            o3d.visualization.draw_geometries(
                                [spine_mesh, geometry])
                            use_cache = True
                            spine_mesh, spine_obj = get_deformed_mesh(
                                spine_mesh, partition_map, spine_spline_indices, curve_surg_points, np.asarray(matching_cloud.points), matching_cloud, use_cache, pcd_spine)

                    # height matching better if only pick points on process
                    if visualize:
                        if first:

                            # vis.add_geometry(geometry)
                            vis.add_geometry(whole_thing)
                            vis.add_geometry(spine_mesh)

                        else:
                            # vis.update_geometry(geometry)
                            vis.add_geometry(whole_thing)
                            vis.update_geometry(spine_mesh)

                        vis.poll_events()
                        vis.update_renderer()

                    else:
                        if view_static:
                            if first:
                                # spheres, arrows = get_joints_with_axes(
                                #     spine_obj)
                                spine_control_points = np.asarray(spine_mesh.vertices)[
                                    spine_spline_indices]

                                rough_correspondence_vertices = np.asarray(spine_mesh.vertices)[
                                    spine_correspondence_indices]

                                clean_mesh(
                                    spine_mesh, partition_map)
                                spine_spline_indices = get_closest_vertices(
                                    spine_control_points, np.asarray(spine_mesh.vertices))
                                spine_correspondence_indices = get_closest_vertices(
                                    rough_correspondence_vertices, np.asarray(spine_mesh.vertices))

                                pcd_spine = spine_mesh.sample_points_uniformly(
                                    number_of_points=10000)
                                pcd_spine = spine_mesh.sample_points_poisson_disk(
                                    number_of_points=3000, pcl=pcd_spine)
                                # o3d.visualization.draw_geometries(
                                #     [*spheres, *arrows, pcd_spine])
                                matching_cloud, matching_indices = find_points_within_radius(
                                    whole_thing, pcd_spine, radius=8)
                                o3d.visualization.draw_geometries(
                                    [spine_mesh, whole_thing])

                                new_whole_thing = copy.deepcopy(whole_thing)
                                new_whole_thing_colors = np.asarray(
                                    new_whole_thing.colors)
                                new_whole_thing_colors[matching_indices] = np.array(
                                    [[255, 165, 0] for _ in range(len(matching_indices))])
                                new_whole_thing.colors = o3d.utility.Vector3dVector(
                                    new_whole_thing_colors)

                    first = False
                    inv_fine_transform = np.linalg.inv(icp_transform)

                    spine_mesh.transform(inv_fine_transform)

                    print(f"frame {frame_index}")
                    rolling_correspondences = ((
                        (rolling_correspondences)*frame_index) + len(np.asarray(evaluation.correspondence_set))) / (frame_index + 1)

                    rolling_rmse = ((
                        (rolling_rmse)*frame_index) + evaluation.inlier_rmse) / (frame_index + 1)
                    print("Correspondences, RMSE")
                    print(rolling_correspondences)
                    print(rolling_rmse)
                    frame_index += 1

            except EOFError:
                break
            except ValueError:
                continue
    finally:
        playback.close()

# "/home/connor/trial_recordings/V6TNrEsM_20240529_155036.mkv"
# "/home/connorscomputer/Downloads/RLqagzpD_20240724_091405.mkv"
# "/home/connorscomputer/Downloads/rRVdsVEp_20240724_091125.mkv"
# "/home/connorscomputer/Downloads/HReJ6KFM_20240916_090806.mkv"
# "/home/connorscomputer/Downloads/HmCerybV_20240920_092421.mkv"
# Mouse clicked at coordinates: (648, 326) Ped L4
# Mouse clicked at coordinates: (608, 324) Ped L5
# Mouse clicked at coordinates: (614, 343) SPL4
# Mouse clicked at coordinates: (653, 345) SPL3
# Mouse clicked at coordinates: (576, 348) SPL5

# HReJ6KFM_20240916_090806 (height of camera: 39 cm)
# pedicle of L4 (minute 0:24)
# pedicle of L5 (minute 0:43) ###maybe don't use this
# spinous process of L4 (minute 0:48)
# spinous process of L3 (in correspondance to the navigation system, minute 0:49)
# spinous process of L5 (minute 0:51)
# try PL4(L), PL4(R), PL4(R), SL4, SL3, SL5


def get_joints_with_axes(spine_obj):
    all_joints = spine_obj.get_all_joints()
    axes = [spine_obj.root_vertebrae.get_joint_axes(
        joint) for joint in all_joints]

    spheres = []
    arrows = []
    for ind, joint in enumerate(all_joints):

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
        spheres.append(sphere.translate(joint.position))
        vectors = axes[ind]

        for ind2, vector in enumerate(vectors):
            color = np.zeros(3)
            color[ind2] = 1
            arrows.append(create_arrow(joint.position,
                          vector, scale=20.0, color=color))
    return spheres, arrows


def get_closest_vertices(spine_control_points, spine_mesh_vertices):
    indices = []
    for point in spine_control_points:
        indices.append(
            np.argmin(np.sqrt(np.sum((spine_mesh_vertices - point)**2, axis=1))))
    return np.array(indices)


def main() -> None:
    parser = ArgumentParser(description="pyk4a player")
    parser.add_argument(
        "--seek", type=float, help="Seek file to specified offset in seconds", default=0.0)

    parser.add_argument('--record-spots', action='store_true', default=False)
    parser.add_argument(
        "FILE", type=str, help="Path to MKV file written by k4arecorder")

    parser.add_argument(
        "--mesh-file", type=str, help="Path to ply file with ground truth"
    )

    parser.add_argument(
        "--set-mesh-points", help="Set mesh points", action='store_true', default=False)

    parser.add_argument('--init-frame', type=int,
                        help="Reference frame", default=0)

    parser.add_argument(
        "--set-spline-curve", help="Set mesh points", action='store_true', default=False)

    args = parser.parse_args()
    filename: str = args.FILE
    offset: float = args.seek
    record_spots = args.record_spots
    mesh_file_path = args.mesh_file
    record_mesh_points = args.set_mesh_points
    record_spline_points = args.set_spline_curve
    mesh_file_path = "/home/connorscomputer/Documents/CT_DATA/CT_NIFTIS/NIFTI_SOLIDS/smoothed_o3d_mask_color.ply"
    baseline_frame = args.init_frame

    playback = PyK4APlayback(filename)

    file_short = filename.split('/')[-1].split('.')[0]
    play(playback, offset, record_spots, file_short,
         mesh_filepath=mesh_file_path, baseline_frame=baseline_frame, record_mesh_points=record_mesh_points, record_spline_points=record_spline_points)


if __name__ == "__main__":
    main()


# test args: --mesh-file /home/connorscomputer/Documents/CT_DATA/CT_NIFTIS/NIFTI_SOLIDS/smoothed_o3d_mask_color.ply
# --file /home/connorscomputer/Downloads/HReJ6KFM_20240916_090806.mkv
