
import cv2
import numpy as np

from pyk4a import PyK4APlayback, ImageFormat
import open3d as o3d
from scipy.spatial import distance_matrix

import pickle

from utils import get_scene_image_from_capture, info, get_color_name, get_scene_geometry_from_capture
from registration import source_icp_transform


COLORS = [(r, g, b) for r in [0, 255] for g in [0, 255] for b in [0, 255]]


def tracker_continue(tracker, new_frame, color, circle_size=5):
    new_frame = new_frame.copy()
    (success, box) = tracker.update(new_frame)
    if success:
        (x, y, w, h) = [int(v) for v in box]
        cv2.circle(new_frame, (x, y), circle_size//2,
                   color, 2)
    return new_frame


def track_on_list(init_frame, init_coord_x, init_coord_y, width, height, remaining_frames, color, circle_size=5):
    tracker = cv2.TrackerCSRT_create()

    init_frame_copy = init_frame.copy()
    # TODO change from same box size
    tracker.init(init_frame_copy, (init_coord_x,
                 init_coord_y, width, height))
    marked_frames = []
    for frame_ind, frame in enumerate(remaining_frames):
        new_frame = frame
        (success, box) = tracker.update(new_frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]

        else:
            x = init_coord_x
            y = init_coord_y
            w = width
            h = height
            tracker.init(remaining_frames[frame_ind-15], (init_coord_x,
                                                          init_coord_y, width, height))

        marked_frames.append((x, y, w, h, color))

    return [(init_coord_x, init_coord_y, width, height, color)] + marked_frames


def write_bounding_boxes(store_frames: list, coordinates: list, frame_no: int):

    marked_coordinates = []

    x_tl, y_tl, color = coordinates[0]
    x_br, y_br, _ = coordinates[1]
    w = x_br - x_tl
    h = y_br - y_tl

    landmark_frame = store_frames[frame_no]
    subsequent_frames = store_frames[frame_no+1:]
    prev_frames = store_frames[:frame_no]

    marked_sub_frames = track_on_list(
        landmark_frame, x_tl, y_tl, w, h, subsequent_frames, color)[1:]

    marked_prev_frames = track_on_list(
        landmark_frame, x_tl, y_tl, w, h, prev_frames[::-1], color)[::-1]

    marked_coordinates = marked_prev_frames + marked_sub_frames
    frame_markings = []
    for ind, frame in enumerate(store_frames):

        for coordinate in marked_coordinates:
            x, y, w, h, color = coordinate
            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          color, 2)
        frame_markings.append(frame)

    return frame_markings, marked_coordinates


def play_written_frames(frames):
    cv2.namedWindow('WrittenImage')
    for frame in frames:

        cv2.imshow("WrittenImage", frame)
        key = cv2.waitKey(int(1000 / 30))


def get_relevent_keypoint_frames(playback: PyK4APlayback):
    frame_cache = []
    times = []
    start_timing = False
    frame_counter = 0
    try:
        playback.open()
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

                    times.append(elapsed)
                    img = capture.color
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    frame_cache.append(img)
                    frame_counter += 1

            except EOFError:
                break
    finally:
        playback.close()
    return frame_cache, times


def mark_relevant_frames(frame_cache, baseline_frame=0):

    coordinates = []
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
            color = COLORS[0]
            coordinates.append((x, y, color))

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', internal_marker_callback)

    for index, rel_frame in enumerate(frame_cache):

        if index == baseline_frame:

            current_keypoint = index

            while True:
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
            break
    cv2.destroyAllWindows()
    return coordinates, baseline_frame


def record_pointed_spots(playback: PyK4APlayback, short_file: str):

    paused = False
    font = cv2.FONT_HERSHEY_SIMPLEX
    cornerLoc = (0, 25)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 2
    lineType = 2

    # this function gets keypoints with second: frame index, frame list, times for each frame
    frame_cache, times = get_relevent_keypoint_frames(
        playback)

    coordinates, baseline_frame = mark_relevant_frames(
        frame_cache)

    written_frames, marked_coordinates = write_bounding_boxes(
        frame_cache, coordinates, baseline_frame)

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

    print('assign bounding box')


def raw_get_3d_coordinates(coordinates_2d: tuple, depth_map: np.array, calibration):
    """
    tuple (x, y)
    numpy depth map
    calibration object
    """
    x, y = coordinates_2d
    ddval = depth_map[y][x]
    return calibration.convert_2d_to_3d(
        (x, y), ddval, 1)


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


def pick_mesh_points(mesh):

    vis = o3d.visualization.VisualizerWithVertexSelection()

    selected = []

    def sel_changed():

        for p in vis.get_picked_points():
            if p.index not in selected:
                selected.append(p.index)
    vis.register_selection_changed_callback(sel_changed)
    vis.create_window()

    # Add the mesh to the visualizer
    vis.add_geometry(mesh)

    vis.run()  # user picks points
    print("")
    vis.destroy_window()
    return selected


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def play_single_initial_frame_mark_both(spine_mesh, playback, offset, baseline_frame, scale_scene=1.0):
    try:
        frame_index = 0
        playback.open()

        info(playback)
        if offset != 0.0:
            playback.seek(int(offset * 1000000))
        picked_points = []
        geom = None
        while True:
            try:
                capture = playback.get_next_capture()
                if capture.color is not None and capture.depth is not None and geom is None:

                    if frame_index == baseline_frame:

                        colors, points = get_scene_geometry_from_capture(
                            capture)
                        geometry = o3d.geometry.PointCloud()
                        geometry.points = o3d.utility.Vector3dVector(
                            points * scale_scene)
                    geometry.colors = o3d.utility.Vector3dVector(
                        (colors/255).astype('float64'))

                    picked_scene_points = pick_points(geometry)
                    scene_picked_points = (
                        points * scale_scene)[picked_scene_points]

                    dist1 = distance_matrix(
                        scene_picked_points, scene_picked_points)

                    picked_points_coords = pick_mesh_points(spine_mesh)
                    picked_points = np.asarray(spine_mesh.vertices)[
                        picked_points_coords]

                    dist2 = distance_matrix(picked_points, picked_points)

                    print(f"max distance scene {np.max(dist1)}")
                    print(f"max distance mesh {np.max(dist2)}")

                    break
                    frame_index += 1

            except EOFError:
                break

    finally:
        playback.close()
        return [picked_points, scene_picked_points]


# this might not be needed
def play_single_initial_frame(spine_mesh, playback, offset, baseline_frame, coordinate_list):
    try:
        frame_index = 0
        playback.open()

        info(playback)
        if offset != 0.0:
            playback.seek(int(offset * 1000000))
        picked_points = []

        while True:
            try:
                capture = playback.get_next_capture()
                if capture.color is not None and capture.depth is not None:

                    if frame_index == baseline_frame:

                        coordinates = coordinate_list[frame_index]
                        circle_size = 5
                        img = get_scene_image_from_capture(capture)
                        cvimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        for coordinate in coordinates:
                            x, y, w, h, color = coordinate
                            shift_x = x + (w//2)
                            shift_y = y + (h//2)
                            cv2.circle(cvimg, (shift_x, shift_y), circle_size//2,
                                       (color[2], color[1], color[0]), 2)

                        color_mappings = {
                            ind+1: get_color_name(x) for ind, (_, _, _, _, x) in enumerate(coordinates)}
                        print(color_mappings)

                        picked_points = pick_mesh_points(spine_mesh, cvimg)

                        break
                    frame_index += 1

            except EOFError:
                break

    finally:
        playback.close()
        return picked_points


def get_scene_geometry_from_capture(capture):
    capture._color = cv2.cvtColor(cv2.imdecode(
        capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
    capture._color_format = ImageFormat.COLOR_BGRA32
    points = capture.transformed_depth_point_cloud.reshape(
        (-1, 3)).astype('float64')
    colors = capture.color[..., (2, 1, 0)].reshape(
        (-1, 3))

    return colors, points


def subsample_mesh(colors, points, bbox_params, calibration, depth_map, offset_y=0, offset_x=0, z_percentile=0.05):

    quantile = np.quantile(points[:, 2], z_percentile)
    mask_z = points[:, 2] > quantile

    min_x = bbox_params[0][0] - offset_x
    min_y = bbox_params[0][1] - offset_y
    max_x = min_x + bbox_params[1] + 2*offset_x
    max_y = min_y + bbox_params[2] + 2*offset_y

    point_coords = [(min_x, min_y),
                    (max_x, max_y)]
    coords_3d = [raw_get_3d_coordinates(x, depth_map, calibration)
                 for x in point_coords]
    min_x_3d, min_y_3d, _ = coords_3d[0]
    max_x_3d, max_y_3d, _ = coords_3d[1]
    mask_x = (points[:, 0] > min_x_3d
              ) & (points[:, 0] < max_x_3d)

    mask_y = (points[:, 1] > min_y_3d
              ) & (points[:, 1] < max_y_3d)

    combined_mask = mask_x & mask_y & mask_z

    return colors[combined_mask], points[combined_mask]
