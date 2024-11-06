from argparse import ArgumentParser
import cv2
import numpy as np
from pyk4a import PyK4APlayback, ImageFormat
import open3d as o3d
import pickle
from typing import Optional
from utils import info, remove_distant_points
from markup_sequence import record_pointed_spots, get_3d_coordinates, raw_get_3d_coordinates, play_single_initial_frame_mark_both
from registration import rough_register_via_correspondences, source_icp_transform, invert_icp_point_to_plane, force_below_z_threshold
from armature_utils import get_joint_positions
from joints import create_armature_objects, Spine
from collections import defaultdict
import copy
import itertools
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


def subsample_mesh(colors, points, bbox_params, calibration, depth_map, offset=0, z_percentile=0.05):
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
    mask_x = (points[:, 0] > (filter_coords[0] - offset)
              ) & (points[:, 0] < (filter_coords[2] + offset))

    mask_y = (points[:, 1] > (filter_coords[1] - offset)
              ) & (points[:, 1] < (filter_coords[3] + offset))

    mask_z = (points[:, 2] < z_cap)

    combined_mask = mask_x & mask_y & mask_z

    return colors[combined_mask], points[combined_mask]


def partition(mesh):
    color_map = defaultdict(list)
    colors = np.asarray(mesh.vertex_colors)
    for ind in range(len(colors)):
        r, g, b = colors[ind]
        color_map[(r, g, b)].append(ind)
    print(len(color_map))
    return color_map


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


def get_deformed_mesh(spine_mesh, partition_map, spine_spline_indices, curve_surg_points, use_cache=False):

    links = get_joint_positions(partition_map, spine_mesh)
    root_node = create_armature_objects(links, partition_map)
    root_node.set_parents()

    # instead, we want catmull rom splines

    spine_obj = Spine(root_node, np.array(
        spine_mesh.vertices), spine_spline_indices, curve_surg_points, alpha=1.0,
        initial_transform=None)

    if use_cache:
        with open(f"fparam_cache.npy", 'rb') as f:
            # source, destination
            fparams = np.load(f)
    else:
        fparams, _ = spine_obj.run_optimization()
        with open(f"fparam_cache.npy", 'wb+') as f:
            # source, destination
            np.save(f, np.array(fparams))

    new_root_node = create_armature_objects(
        links, partition_map)
    new_root_node.set_parents()
    spine_obj_original = Spine(new_root_node, np.array(
        spine_mesh.vertices), spine_spline_indices, curve_surg_points, alpha=1.0)
    spine_obj_original.apply_joint_parameters(fparams)
    spine_obj_original.apply_joint_angles()

    new_mesh = copy.deepcopy(spine_mesh)
    new_mesh.vertices = o3d.utility.Vector3dVector(
        spine_obj_original.vertices)

    new_mesh.compute_vertex_normals()  # Recompute vertex normals
    new_mesh.compute_triangle_normals()
    # clean_mesh(new_mesh, partition_map)

    return new_mesh


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

    curve_surg_points = fit_catmull_rom(surg_points_spline)
    # I think the ordering of this is wrong
    # curve_spine_points = fit_catmull_rom(
    #     remove_distant_points(spine_points_spline))

    # number of joints is number of vertebrae -1
    # number of bezier control points = number of joints + 2
    curve_surg_points_for_icp = fit_catmull_rom(
        surg_points_spline, sample_points=5000)
    curve_surge_pcd = o3d.geometry.PointCloud()
    curve_surge_pcd.points = o3d.utility.Vector3dVector(
        curve_surg_points_for_icp)

    partition_map = partition(spine_mesh)

    rgb_num_map = {key: ind for ind, key in enumerate(partition_map)}
    # replace keys in partition maps with
    for key, value in rgb_num_map.items():
        partition_map[value] = partition_map.pop(key)

    # clean_mesh(spine_mesh, partition_map)

    source_points = pp_list[0]
    scene_points = pp_list[1]
    rough_transform = get_rough_transform(
        source_points, scene_points, scale_scene)

    spine_spline_indices = get_closest_spine_indices(
        spine_points_spline, spine_mesh)

    geometry = o3d.geometry.PointCloud()

    frame_index = 0

    first = True

    # pcd for mesh from CT - this is correct scale, camera needs scaling
    pcd_spine = o3d.geometry.PointCloud()
    pcd_spine.points = spine_mesh.vertices
    pcd_spine.normals = spine_mesh.vertex_normals

    identity_transform = np.array([[1., 0., 0., 0],
                                   [0., 1., 0., 0],
                                   [0., 0., 1., 0],
                                   [0., 0., 0., 1.]])

    # do registration for 1st

    visualize = True

    if visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
    deformed_mesh = None
    try:
        playback.open()

        info(playback)
        if offset != 0.0:
            playback.seek(int(offset * 1000000))
        while True:
            try:
                capture = playback.get_next_capture()
                if capture.color is not None and capture.depth is not None:
                    transforms = []
                    inverse_transforms = []
                    x_c, y_c, w_c, h_c, _ = coordinates[frame_index]
                    colors, points = get_scene_geometry_from_capture(capture)
                    subsample_colors, subsample_points = subsample_mesh(
                        colors, points, ((x_c, y_c), w_c, h_c), capture._calibration, capture.transformed_depth, offset=20.0, z_percentile=0.05)

                    geometry.points = o3d.utility.Vector3dVector(
                        subsample_points)
                    geometry.colors = o3d.utility.Vector3dVector(
                        (subsample_colors/255).astype('float64'))

                    curve_transform = source_icp_transform(
                        curve_surge_pcd, geometry, identity_transform, threshold=0.2
                    )

                    curve_surge_pcd.transform(curve_transform)

                    geometry.points.extend(curve_surge_pcd.points)

                    icp_transform = source_icp_transform(
                        pcd_spine, geometry, rough_transform, threshold=0.4)

                    # icp_transform = invert_icp_point_to_plane(
                    #     pcd_spine, geometry, rough_transform, threshold=0.5)

                    spine_mesh.transform(icp_transform)
                    # print(icp_transform)
                    if first:
                        spine_mesh = get_deformed_mesh(
                            spine_mesh, partition_map, spine_spline_indices, curve_surg_points, use_cache=True)

                    z_transform = force_below_z_threshold(
                        spine_mesh, curve_surg_points)
                    inverse_z_transform = np.linalg.inv(z_transform)

                    spine_mesh.transform(z_transform)
                    pcd_sample = spine_mesh.sample_points_uniformly(
                        number_of_points=10000)
                    icp_transform2 = source_icp_transform(
                        pcd_sample, geometry, np.eye(4), threshold=0.4)
                    spine_mesh.transform(icp_transform2)

                    inv_fine_transform = np.linalg.inv(icp_transform)
                    inv_curve_transform = np.linalg.inv(curve_transform)
                    inv_transform2 = np.linalg.inv(icp_transform2)
                    curve_surge_pcd.transform(inv_curve_transform)
                    if visualize:
                        if first:

                            vis.add_geometry(geometry)
                            vis.add_geometry(spine_mesh)
                            first = False
                        else:
                            vis.update_geometry(geometry)
                            vis.update_geometry(spine_mesh)

                        vis.poll_events()
                        vis.update_renderer()
                        frame_index += 1
                    else:
                        o3d.visualization.draw_geometries(
                            [geometry, spine_mesh])
                        # o3d.visualization.draw_geometries(
                        #     [source_whole_rough, source_scene_rough])
                    # spine_mesh.transform(inv_fine_transform)
                    spine_mesh.transform(inv_transform2)
                    spine_mesh.transform(inverse_z_transform)
                    spine_mesh.transform(inv_fine_transform)
                    if first:
                        spine_spline_points = fit_catmull_rom(np.asarray(spine_mesh.vertices)[
                            spine_spline_indices], sample_points=5000)

                        pcd_spine = o3d.geometry.PointCloud()
                        pcd_spine.points = spine_mesh.vertices
                        pcd_spine.normals = spine_mesh.vertex_normals
                        pcd_spine.points.extend(spine_spline_points)

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
