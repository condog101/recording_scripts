from argparse import ArgumentParser
import cv2
import numpy as np
from pyk4a import PyK4APlayback, ImageFormat
import open3d as o3d

import pickle
from typing import Optional
from utils import process_time_string, info
from markup_sequence import record_pointed_spots, create_sphere_at_coordinate, get_3d_coordinates, pick_mesh_points, raw_get_3d_coordinates, play_single_initial_frame
from registration import rough_register_via_correspondences


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


def create_and_update_vis_with_spheres(frame_3d_coordinates, vis):
    spheres = [create_sphere_at_coordinate(
        x[0], 3, color=np.array(x[3])/255) for x in frame_3d_coordinates]

    for ind in range(len(spheres)):
        vis.add_geometry(spheres[ind])

    return spheres


def subsample_mesh(colors, points, bbox_coords, calibration, depth_map):
    bbox_3d = [raw_get_3d_coordinates(
        coord, depth_map, calibration) for coord in bbox_coords]

    min_x = bbox_coords[0][0]
    min_y = bbox_coords[0][1]
    max_x = bbox_coords[1][0]
    max_y = bbox_coords[1][1]


def play(playback: PyK4APlayback, offset: float, record_spots: bool, file_short: str, keypoints: list, mesh_filepath: Optional[str] = None,
         baseline_frame: int = 0, record_mesh_points: bool = False, scale_scene=10.0):

    if record_spots:
        record_pointed_spots(playback, file_short, keypoints)
    bbox_file = f"bbox_coords_{file_short}.npy"

    coordinates_file = f"stored_coordinates_{file_short}.pkl"
    time_file = f"times_{file_short}.pkl"

    with open(bbox_file, 'rb') as f:
        # source, destination
        bbox_coords = np.load(f)

    with open(coordinates_file, 'rb') as f:
        coordinates = pickle.load(f)

    with open(time_file, 'rb') as f:
        times = pickle.load(f)

    coordinate_list = list(zip(*coordinates.values()))

    spine_mesh = o3d.io.read_triangle_mesh(mesh_filepath)
    if spine_mesh is not None and record_mesh_points:
        picked_points = play_single_initial_frame(
            spine_mesh, playback, offset, baseline_frame, coordinate_list)
        pp_list = np.array([x.coord for x in picked_points])
        print(pp_list)
        with open(f"meshpoints_{file_short}.npy", 'wb+') as f:
            # source, destination
            np.save(f, pp_list)
    else:
        with open(f"meshpoints_{file_short}.npy", 'rb') as f:
            # source, destination
            pp_list = np.load(f)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()

    frame_index = 0

    first = True

    pcd_spine = o3d.geometry.PointCloud()
    pcd_spine.points = spine_mesh.vertices

    source_whole_rough = o3d.geometry.PointCloud()
    source_whole_rough.points = o3d.utility.Vector3dVector(pp_list)

    try:
        playback.open()

        info(playback)
        if offset != 0.0:
            playback.seek(int(offset * 1000000))
        while True:
            try:
                capture = playback.get_next_capture()
                if capture.color is not None and capture.depth is not None:

                    colors, points = get_scene_geometry_from_capture(capture)
                    subsample_colors, subsample_points = subsample_mesh(
                        colors, points, bbox_coords, capture._calibration, capture.transformed_depth)
                    geometry.points = o3d.utility.Vector3dVector(
                        points * scale_scene)
                    geometry.colors = o3d.utility.Vector3dVector(
                        (colors/255).astype('float64'))

                    frame_3d_coordinates = get_3d_coords_from_stored_2d(
                        coordinate_list, frame_index, capture)
                    rough_transform = rough_register_via_correspondences(source_whole_rough, [
                        np.array(x[0]) * scale_scene for x in frame_3d_coordinates])

                    inv_rough_transform = np.linalg.inv(rough_transform)
                    spine_mesh.transform(rough_transform)
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
                    spine_mesh.transform(inv_rough_transform)

            except EOFError:
                break
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
# PL4, try PL3, SL4, SL3, SL5


def add_mesh_to_scene(mesh_file_path, playback, initial_frame):
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)


def main() -> None:
    parser = ArgumentParser(description="pyk4a player")
    parser.add_argument(
        "--seek", type=float, help="Seek file to specified offset in seconds", default=0.0)

    parser.add_argument('--record-spots', action='store_true', default=False)
    parser.add_argument(
        "FILE", type=str, help="Path to MKV file written by k4arecorder")
    parser.add_argument(
        "--keypoints", type=str, help="keyframes (seconds)", default="")

    parser.add_argument(
        "--mesh-file", type=str, help="Path to ply file with ground truth"
    )

    parser.add_argument(
        "--set-mesh-points", help="Set mesh points", action='store_true', default=False)

    parser.add_argument('--init-frame', type=int,
                        help="Reference frame", default=0)

    args = parser.parse_args()
    filename: str = args.FILE
    offset: float = args.seek
    record_spots = args.record_spots
    mesh_file_path = args.mesh_file
    record_mesh_points = args.set_mesh_points
    mesh_file_path = "/home/connorscomputer/Documents/CT_DATA/CT_NIFTIS/NIFTI_SOLIDS/smoothed_o3d_mask_color.ply"
    baseline_frame = args.init_frame

    keypoints = process_time_string(args.keypoints)

    if mesh_file_path is not None:
        print("Adding mesh to frame {init_frame}")

    playback = PyK4APlayback(filename)

    file_short = filename.split('/')[-1].split('.')[0]
    play(playback, offset, record_spots, file_short, keypoints,
         mesh_filepath=mesh_file_path, baseline_frame=baseline_frame, record_mesh_points=record_mesh_points)


if __name__ == "__main__":
    main()


# test args: --mesh-file /home/connorscomputer/Documents/CT_DATA/CT_NIFTIS/NIFTI_SOLIDS/smoothed_o3d_mask_color.ply
# --file /home/connorscomputer/Downloads/HReJ6KFM_20240916_090806.mkv
