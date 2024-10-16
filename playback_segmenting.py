from argparse import ArgumentParser
import cv2
import numpy as np
from typing import Optional, Tuple
from pyk4a import PyK4APlayback, ImageFormat
import open3d as o3d

import pickle


from utils import process_time_string
from markup_sequence import record_pointed_spots, create_sphere_at_coordinate, get_3d_coordinates, update_sphere_position


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


def play(playback: PyK4APlayback, record_spots: bool, file_short: str, keypoints: list):

    if record_spots:
        record_pointed_spots(playback, file_short, keypoints)

    coordinates_file = f"stored_coordinates_{file_short}.pkl"
    time_file = f"times_{file_short}.pkl"
    # frames_file = f"output_{file_short}.mp4"

    with open(coordinates_file, 'rb') as f:
        coordinates = pickle.load(f)

    with open(time_file, 'rb') as f:
        times = pickle.load(f)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    first = True
    frame_index = 0

    coordinate_list = list(zip(*coordinates.values()))
    spheres = []

    while True:
        try:
            capture = playback.get_next_capture()
            if capture.color is not None and capture.depth is not None:

                capture._color = cv2.cvtColor(cv2.imdecode(
                    capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
                capture._color_format = ImageFormat.COLOR_BGRA32
                points = capture.transformed_depth_point_cloud.reshape(
                    (-1, 3)).astype('float64')
                colors = capture.color[..., (2, 1, 0)].reshape(
                    (-1, 3))
                geometry.points = o3d.utility.Vector3dVector(points)
                geometry.colors = o3d.utility.Vector3dVector(
                    (colors/255).astype('float64'))

                frame_2d_coordinates = coordinate_list[frame_index]
                frame_3d_coordinates = get_3d_coordinates(
                    frame_2d_coordinates, capture.transformed_depth, capture._calibration)

                if first:
                    spheres = [create_sphere_at_coordinate(
                        x[0], 3, color=x[3]) for x in frame_3d_coordinates]
                    vis.add_geometry(geometry)
                    for ind in range(len(spheres)):
                        vis.add_geometry(spheres[ind])
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
# pedicle of L5 (minute 0:43)
# spinous process of L4 (minute 0:48)
# spinous process of L3 (in correspondance to the navigation system, minute 0:49)
# spinous process of L5 (minute 0:51)


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
        "--mesh_file", help="Path to ply file with ground truth"
    )
    parser.add_argument('--playback', action='store_true', default=False)

    args = parser.parse_args()
    filename: str = args.FILE
    offset: float = args.seek
    record_spots = args.record_spots
    mesh_file_path = args.mesh_file
    view_markings = args.playback

    keypoints = process_time_string(args.keypoints)

    if view_markings:
        playback = PyK4APlayback(filename)
        playback.open()

        info(playback)

        if offset != 0.0:
            playback.seek(int(offset * 1000000))
        file_short = filename.split('/')[-1].split('.')[0]
        play(playback, record_spots, file_short, keypoints)

        playback.close()

    if mesh_file_path is not None:
        add_mesh_to_scene(mesh_file_path)


if __name__ == "__main__":
    main()
