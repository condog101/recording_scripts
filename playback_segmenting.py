from argparse import ArgumentParser
import cv2
import numpy as np
from typing import Optional, Tuple
from pyk4a import PyK4APlayback, ImageFormat
import open3d as o3d
import curses
import time
import sys
import tty
import os
import termios
import pickle
import multiprocessing
from queue import Empty
from utils import create_bounding_box, process_time_string, is_closest_to_keypoints


def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])  # type: ignore
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


COLORS = [(r, g, b) for r in [0, 255] for g in [0, 255] for b in [0, 255]]


def lanczos_resize(img, scale_factor):
    # resize img by a scale_factor using Lanczos interpolation
    return cv2.resize(img, None,
                      fx=scale_factor, fy=scale_factor,
                      interpolation=cv2.INTER_LANCZOS4)


def laplacian_filter(img):
    # Laplacian sharpening filter
    return cv2.Laplacian(img, -1, ksize=5, scale=1,
                         delta=0, borderType=cv2.BORDER_DEFAULT)


def tracker_continue(tracker, new_frame, color, circle_size=5):
    new_frame = new_frame.copy()
    (success, box) = tracker.update(new_frame)
    if success:
        (x, y, w, h) = [int(v) for v in box]
        cv2.circle(new_frame, (x, y), circle_size//2,
                   color, 2)
    return new_frame


def track_on_list(init_frame, init_coord_x, init_coord_y, box_size, remaining_frames, color, circle_size=5):
    tracker = cv2.TrackerCSRT_create()
    # tracker = cv2.legacy.TrackerMedianFlow_create()
    init_frame_copy = init_frame.copy()
    tracker.init(init_frame_copy, (init_coord_x,
                 init_coord_y, box_size, box_size))
    marked_frames = []
    for frame_ind, frame in enumerate(remaining_frames):
        new_frame = frame
        (success, box) = tracker.update(new_frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            # cv2.circle(new_frame, (x + (w//2), y + (h//2)), circle_size//2,
            #            color, 2)
        else:
            x = init_coord_x
            y = init_coord_y
            w = box_size
            h = box_size
            tracker.init(remaining_frames[frame_ind-15], (init_coord_x,
                                                          init_coord_y, box_size, box_size))
            # cv2.circle(new_frame, (init_coord_x + (box_size//2), init_coord_y + (box_size//2)), circle_size//2,
            #            color, 2)

        marked_frames.append((x, y, w, h, color))
    # cv2.circle(init_frame_copy, (init_coord_x + (box_size//2), init_coord_y + (box_size//2)), circle_size//2,
    #            color, 2)

    return [(init_coord_x, init_coord_y, box_size, box_size, color)] + marked_frames


def write_bounding_boxes(store_frames: list, coordinates: dict, box_size=15, circle_size=5):

    marked_coordinates = {}
    for ind, coordinates in coordinates.items():
        x_c, y_c, color = coordinates

        frame_no = ind

        landmark_frame = store_frames[frame_no]
        subsequent_frames = store_frames[frame_no+1:]
        prev_frames = store_frames[:frame_no]
        x_c, y_c, _, _ = create_bounding_box(
            x_c-(box_size//2), y_c-(box_size//2), box_size=box_size)
        marked_sub_frames = track_on_list(
            landmark_frame, x_c, y_c, box_size, subsequent_frames, color)[1:]

        marked_prev_frames = track_on_list(
            landmark_frame, x_c, y_c, box_size, prev_frames[::-1], color)[::-1]

        marked_coordinates[frame_no] = marked_prev_frames + marked_sub_frames
    frame_markings = []
    for ind, frame in enumerate(store_frames):

        for key, coordinate_set in marked_coordinates.items():
            x, y, w, h, color = coordinate_set[ind]
            cv2.circle(frame, (x + (w//2), y + (h//2)), circle_size//2,
                       color, 2)
        frame_markings.append(frame)

    return frame_markings, marked_coordinates


def play_written_frames(frames):
    cv2.namedWindow('WrittenImage')
    for frame in frames:

        cv2.imshow("WrittenImage", frame)
        key = cv2.waitKey(int(1000 / 30))


def get_relevent_keypoint_frames(playback: PyK4APlayback, keypoints: list):
    keypoint_dict = {}
    frame_cache = []
    times = []
    start_timing = False
    frame_counter = 0
    while True:
        try:
            capture = playback.get_next_capture()

            if capture.color is not None and capture.depth is not None:

                capture._color = cv2.cvtColor(cv2.imdecode(
                    capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
                capture._color_format = ImageFormat.COLOR_BGRA32
                if not start_timing:
                    start_timing = True
                    start_time = capture._color_timestamp_usec
                    elapsed = 0.0
                else:
                    elapsed = round(
                        (capture._color_timestamp_usec - start_time)/1000000, 2)

                closest_keypoint = is_closest_to_keypoints(
                    keypoints, keypoint_dict, elapsed)

                if closest_keypoint is not None:
                    keypoint_dict[closest_keypoint] = frame_counter

                times.append(elapsed)
                img = capture.color
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                frame_cache.append(img)
                frame_counter += 1

        except EOFError:
            break
    return keypoint_dict, frame_cache, times


def mark_relevant_frames(keypoint_dict, frame_cache, times):

    coordinates = {}
    current_keypoint = None
    current_i = None

    window_name = None
    show = True

    def internal_marker_callback(event, x, y, flags, param):

        nonlocal coordinates
        nonlocal current_i
        nonlocal current_keypoint
        nonlocal window_name
        nonlocal show

        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Mouse clicked at coordinates: ({x}, {y})")
            color = COLORS[current_i]
            coordinates[current_keypoint] = (x, y, color)
            show = False

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', internal_marker_callback)

    for i, (sec, index) in enumerate(keypoint_dict.items()):
        current_i = i

        current_keypoint = index
        window_name = f"image_{sec}"
        time_frame = times[index]
        print(time_frame)

        while True:
            rel_frame = frame_cache[current_keypoint]
            cv2.imshow('Image', rel_frame)

            # Wait for 1ms and check for click or key press
            key = cv2.waitKey(1) & 0xFF

            if not show or key == ord('q'):
                break
            elif key == 81 or key == 2424832:  # Left arrow key
                current_keypoint -= 5
                # Add your left arrow key action here
            elif key == 83 or key == 2555904:  # Right arrow key
                current_keypoint += 5

        show = True
    cv2.destroyAllWindows()
    return coordinates


def record_pointed_spots(playback: PyK4APlayback, short_file: str, keypoints: list):

    paused = False
    font = cv2.FONT_HERSHEY_SIMPLEX
    cornerLoc = (0, 25)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 2
    lineType = 2
    # this function gets keypoints with second: frame index, frame list, times for each frame
    keypoint_dict, frame_cache, times = get_relevent_keypoint_frames(
        playback, keypoints)

    coordinates = mark_relevant_frames(keypoint_dict, frame_cache, times)

    written_frames, marked_coordinates = write_bounding_boxes(
        frame_cache, coordinates)

    cv2.namedWindow('Image')

    for frame_ind, img in enumerate(written_frames):
        try:

            if paused:
                key = cv2.waitKey(10)
                if key == ord('x'):
                    paused = not paused
                continue

            elapsed = times[frame_ind]

            t_str = "{:0.2f} Seconds".format(elapsed)

            cv2.putText(img, t_str,
                        cornerLoc,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            cv2.imshow("Image", img)

            key = cv2.waitKey(int(1000 / 25))
            if key == ord('x'):
                paused = not paused

        except EOFError:
            break

    # loop through first, store frames etc

    with open(f"stored_coordinates_{short_file}.pkl", 'wb+') as f:
        # source, destination
        pickle.dump(marked_coordinates, f)

    with open(f"times_{short_file}.pkl", 'wb+') as f:
        # source, destination
        pickle.dump(times, f)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    fps = 30.0  # frames per second
    frame_size = (written_frames[0].shape[1],
                  written_frames[0].shape[0])  # width, height
    out = cv2.VideoWriter(f"output_{short_file}.mp4", fourcc, fps, frame_size)
    for frame in written_frames:
        out.write(frame)

    # Release everything when job is finished
    out.release()


def create_sphere_at_coordinate(center, radius, color=[1, 0, 0]):
    """
    Create a sphere at a specific coordinate.

    :param center: List or numpy array of [x, y, z] coordinates
    :param radius: Radius of the sphere
    :param color: RGB color of the sphere (default is red)
    :return: Open3D sphere geometry
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(center)
    return sphere


def get_3d_coordinates(coordinates_2d, depth_map, calibration):
    points = []
    for coordinate in coordinates_2d:
        x, y, w, h, color = coordinate
        shift_x = x + (w//2)
        shift_y = y + (h//2)
        ddval = depth_map[shift_y][shift_x]
        points.append((calibration.convert_2d_to_3d(
            (shift_x, shift_y), ddval, 1), w, h, color))
    return points


def update_sphere_position(sphere, new_center):
    translation = np.array(new_center) - np.array(sphere.get_center())
    sphere.translate(translation)
    return np.array(new_center)


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
    num_points = len(coordinate_list[0])
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

    args = parser.parse_args()
    filename: str = args.FILE
    offset: float = args.seek
    record_spots = args.record_spots
    keypoints = process_time_string(args.keypoints)

    playback = PyK4APlayback(filename)
    playback.open()

    info(playback)

    if offset != 0.0:
        playback.seek(int(offset * 1000000))
    file_short = filename.split('/')[-1].split('.')[0]
    play(playback, record_spots, file_short, keypoints)

    playback.close()


if __name__ == "__main__":
    main()
