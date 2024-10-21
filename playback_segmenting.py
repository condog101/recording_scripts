from argparse import ArgumentParser
import cv2
import numpy as np
from pyk4a import PyK4APlayback, ImageFormat
import open3d as o3d
import webcolors
import pickle
from typing import Optional
from utils import process_time_string
from markup_sequence import record_pointed_spots, create_sphere_at_coordinate, get_3d_coordinates, update_sphere_position


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def get_color_name(rgb_tuple):
    try:
        # Convert RGB to hex
        hex_value = webcolors.rgb_to_hex(rgb_tuple)
        # Get the color name directly
        return webcolors.hex_to_name(hex_value)
    except ValueError:
        # If exact match not found, find the closest color
        return closest_color(rgb_tuple)


def get_scene_geometry_from_capture(capture):
    capture._color = cv2.cvtColor(cv2.imdecode(
        capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
    capture._color_format = ImageFormat.COLOR_BGRA32
    points = capture.transformed_depth_point_cloud.reshape(
        (-1, 3)).astype('float64')
    colors = capture.color[..., (2, 1, 0)].reshape(
        (-1, 3))
    # points = capture.depth_point_cloud.reshape(
    #     (-1, 3)).astype('float64')
    # colors = capture.transformed_color[..., (2, 1, 0)].reshape(
    #     (-1, 3))
    return colors, points


def get_3d_coords_from_stored_2d(coordinate_list, frame_index, capture):
    frame_2d_coordinates = coordinate_list[frame_index]
    frame_3d_coordinates = get_3d_coordinates(
        frame_2d_coordinates, capture.transformed_depth, capture._calibration)
    return frame_3d_coordinates


def create_and_update_vis_with_spheres(frame_3d_coordinates, vis):
    spheres = [create_sphere_at_coordinate(
        x[0], 3, color=x[3]) for x in frame_3d_coordinates]

    for ind in range(len(spheres)):
        vis.add_geometry(spheres[ind])


def pick_mesh_points(scene, mesh, spheres):

    # vis = o3d.visualization.VisualizerWithEditing()
    # vis.create_window()
    # vis.add_geometry(scene)
    # for ind in range(len(spheres)):
    #     vis.add_geometry(spheres[ind])

    # vis.add_geometry(mesh)
    # vis.run()  # user picks points
    # vis.destroy_window()
    # print("")
    # return vis.get_picked_points()
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the mesh to the visualizer
    vis.add_geometry(mesh)
    vis.add_geometry(scene)

    for ind in range(len(spheres)):
        vis.add_geometry(spheres[ind])

    # Run the visualizer
    vis.run()

    # Close the window
    vis.destroy_window()


def play_single_initial_frame(spine_mesh, playback, offset, baseline_frame, coordinate_list):
    try:
        frame_index = 0
        playback.open()

        info(playback)
        if offset != 0.0:
            playback.seek(int(offset * 1000000))

        geometry = o3d.geometry.PointCloud()

        while True:
            try:
                capture = playback.get_next_capture()
                if capture.color is not None and capture.depth is not None:

                    if frame_index == baseline_frame:

                        frame_3d_coordinates = get_3d_coords_from_stored_2d(
                            coordinate_list, frame_index, capture)

                        colors, points = get_scene_geometry_from_capture(
                            capture)
                        geometry.points = o3d.utility.Vector3dVector(points*10)
                        geometry.colors = o3d.utility.Vector3dVector(
                            (colors/255).astype('float64'))
                        geometry.estimate_normals()

                        spheres = [create_sphere_at_coordinate(
                            x[0], 3, color=np.array(x[3])/255, scaling_factor=10.0) for x in frame_3d_coordinates]

                        sphere_colors = [x[3] for x in frame_3d_coordinates]
                        color_mappings = {
                            ind+1: get_color_name(x) for ind, x in enumerate(sphere_colors)}
                        print(color_mappings)
                        pick_mesh_points(geometry, spine_mesh, spheres)

                        break
                    frame_index += 1

            except EOFError:
                break

    finally:
        playback.close()


def play(playback: PyK4APlayback, offset: float, record_spots: bool, file_short: str, keypoints: list, mesh_filepath: Optional[str] = None, baseline_frame: int = 0):

    if record_spots:
        record_pointed_spots(playback, file_short, keypoints)

    coordinates_file = f"stored_coordinates_{file_short}.pkl"
    time_file = f"times_{file_short}.pkl"

    with open(coordinates_file, 'rb') as f:
        coordinates = pickle.load(f)

    with open(time_file, 'rb') as f:
        times = pickle.load(f)

    coordinate_list = list(zip(*coordinates.values()))

    spine_mesh = o3d.io.read_triangle_mesh(mesh_filepath)
    if spine_mesh is not None:
        play_single_initial_frame(
            spine_mesh, playback, offset, baseline_frame, coordinate_list)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()

    frame_index = 0

    spheres = []
    first = True

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
                    geometry.points = o3d.utility.Vector3dVector(points)
                    geometry.colors = o3d.utility.Vector3dVector(
                        (colors/255).astype('float64'))

                    frame_3d_coordinates = get_3d_coords_from_stored_2d(
                        coordinate_list, frame_index, capture)
                    if first:
                        create_and_update_vis_with_spheres(
                            frame_3d_coordinates, vis)
                        vis.add_geometry(geometry)
                        first = False
                    else:
                        vis.update_geometry(geometry)
                        for ind, coord in enumerate(frame_3d_coordinates):

                            update_sphere_position(spheres[ind], coord[0])
                            vis.update_geometry(spheres[ind])
                    vis.poll_events()
                    vis.update_renderer()
                    frame_index += 1

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
# PL4, SL4, SL3, SL5


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
    parser.add_argument('--init-frame', type=int,
                        help="Reference frame", default=0)

    args = parser.parse_args()
    filename: str = args.FILE
    offset: float = args.seek
    record_spots = args.record_spots
    mesh_file_path = args.mesh_file
    mesh_file_path = "/home/connorscomputer/Documents/CT_DATA/CT_NIFTIS/NIFTI_SOLIDS/smoothed_o3d_mask_color.ply"
    baseline_frame = args.init_frame

    keypoints = process_time_string(args.keypoints)

    if mesh_file_path is not None:
        print("Adding mesh to frame {init_frame}")

    playback = PyK4APlayback(filename)

    file_short = filename.split('/')[-1].split('.')[0]
    play(playback, offset, record_spots, file_short, keypoints,
         mesh_filepath=mesh_file_path, baseline_frame=baseline_frame)


if __name__ == "__main__":
    main()


# test args: --mesh-file /home/connorscomputer/Documents/CT_DATA/CT_NIFTIS/NIFTI_SOLIDS/smoothed_o3d_mask_color.ply
# --file /home/connorscomputer/Downloads/HReJ6KFM_20240916_090806.mkv
