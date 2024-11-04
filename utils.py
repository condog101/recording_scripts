from pyk4a import ImageFormat, PyK4APlayback
import cv2
import webcolors
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def create_bounding_box(top_left_x, top_left_y, width, height):

    bottom_right_x = top_left_x + width
    bottom_right_y = top_left_y + height

    return (top_left_x, top_left_y, bottom_right_x, bottom_right_y)


def process_time_string(keypoints: str):
    try:
        return [int(x.strip()) for x in keypoints.split(',')]
    except ValueError:
        print("Could not parse keypoints, returning empty list")
        return []


def is_closest_to_keypoints(keypoints: list, keypoint_dict: dict, time: float):
    for keypoint in keypoints:
        if keypoint not in keypoint_dict:
            int_time = int(time)
            if int_time == keypoint:
                return keypoint
    return None


def get_scene_image_from_capture(capture):
    capture._color = cv2.cvtColor(cv2.imdecode(
        capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
    capture._color_format = ImageFormat.COLOR_BGRA32

    colors = capture.color[..., (2, 1, 0)]
    return colors


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

    return colors, points


def vector_to_translation_matrix(translation):
    """Convert a 3D translation vector to a 4x4 homogeneous matrix"""
    matrix = np.eye(4)  # Create 4x4 identity matrix
    matrix[:3, 3] = translation  # Set translation components
    return matrix


def emd(X, Y):
    d = cdist(X, Y)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / min(len(X), len(Y))


def clamp_quaternion(quat):
    """
    Clamp quaternion to represent at most 90 degrees rotation in any axis.

    Args:
        quat: numpy array [w, x, y, z] where w is the scalar component

    Returns:
        clamped quaternion
    """
    # Ensure w is positive (this makes the angle interpretation simpler)
    if quat[0] < 0:
        quat = -quat

    # For 90 degrees, w component should be cos(45°) = 1/√2 ≈ 0.7071
    # Because quaternion elements represent half angles
    min_w = np.sqrt(2) / 2

    if quat[0] < min_w:
        # Quaternion represents rotation > 90 degrees
        # Scale vector part to maintain direction but reduce magnitude
        vec_part = quat[1:]
        vec_magnitude = np.linalg.norm(vec_part)
        if vec_magnitude > 0:
            # Calculate new vector magnitude for 90 degree rotation
            new_vec_magnitude = np.sqrt(1 - min_w**2)
            vec_part = (vec_part / vec_magnitude) * new_vec_magnitude
            quat = np.array([min_w, vec_part[0], vec_part[1], vec_part[2]])

    return quat


def normalize_list_of_quaternion_params(quatlist):
    new_params = quatlist.copy()
    for i in range(0, len(quatlist), 4):
        new_params[i:i+4] = clamp_quaternion(quatlist[i:i+4] /
                                             np.linalg.norm(quatlist[i:i+4]))
    return new_params


def remove_distant_points(points, threshold_factor=2.0):
    """
    Remove points that are unusually far from their nearest neighbors.

    Args:
        points: numpy array of shape (n, 3) containing 3D coordinates
        threshold_factor: points with min distance > threshold_factor * mean_min_distance
                        will be considered outliers

    Returns:
        numpy array with outliers removed
    """
    # Calculate pairwise distances between all points
    distances = np.sqrt(((points[:, np.newaxis] - points) ** 2).sum(axis=2))

    # Set diagonal to infinity so we don't count distance to self
    np.fill_diagonal(distances, np.inf)

    # Find minimum distance for each point to any other point
    min_distances = np.min(distances, axis=1)

    # Calculate threshold based on mean of minimum distances
    threshold = np.mean(min_distances) * threshold_factor

    # Keep points whose minimum distance is below threshold
    mask = min_distances < threshold
    return points[mask]


def align_centroids(points1, points2):
    """
    Translate points2 to align its centroid with points1's centroid.

    Args:
        points1: numpy array of shape (n, 3) containing first set of points
        points2: numpy array of shape (m, 3) containing second set of points

    Returns:
        translated_points2: numpy array of shape (m, 3) with aligned centroid
    """
    # Compute centroids
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)

    # Compute translation vector
    translation = centroid1 - centroid2

    # Apply translation to points2
    translated_points2 = points2 + translation

    return translated_points2
