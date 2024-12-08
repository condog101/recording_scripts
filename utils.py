from pyk4a import ImageFormat, PyK4APlayback
import cv2
import webcolors
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# X_LIMIT, Y_LIMIT, Z_LIMIT = 0.2, 1, 1
X_LIMIT, Y_LIMIT, Z_LIMIT = 3, 8, 15


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
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
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


def normalize_list_of_quaternion_params(quatlist):
    new_params = quatlist.copy()
    for i in range(0, len(quatlist), 4):
        new_params[i:i +
                   4], _ = clamp_quaternion_rotation_by_axis(quatlist[i:i+4])
    return new_params


def filter_z_offset_points(points, threshold_factor=2.0):
    z_points = points[:, 2]
    a, b = np.meshgrid(z_points, z_points)
    distances = np.abs(a - b)
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    threshold = np.mean(min_distances) * threshold_factor

    # Keep points whose minimum distance is below threshold
    mask = min_distances < threshold

    return points[mask]


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


def partition(mesh):
    color_map = defaultdict(list)
    colors = np.asarray(mesh.vertex_colors)
    for ind in range(len(colors)):
        r, g, b = colors[ind]
        color_map[(r, g, b)].append(ind)

    return color_map


def average_quaternions(quaternions):
    """
    Calculate average quaternion

    :params quaternions: is a Nx4 numpy matrix and contains the quaternions
        to average in the rows.
        The quaternions are arranged as (w,x,y,z), with w being the scalar

    :returns: the average quaternion of the input. Note that the signs
        of the output quaternion can be reversed, since q and -q
        describe the same orientation
    """

    # Number of quaternions to average
    samples = quaternions.shape[0]
    mat_a = np.zeros(shape=(4, 4), dtype=np.float64)

    for i in range(0, samples):
        quat = quaternions[i, :]
        # multiply quat with its transposed version quat' and add mat_a
        mat_a = np.outer(quat, quat) + mat_a

    # scale
    mat_a = (1.0 / samples)*mat_a
    # compute eigenvalues and -vectors
    eigen_values, eigen_vectors = np.linalg.eig(mat_a)
    # Sort by largest eigenvalue
    eigen_vectors = eigen_vectors[:, eigen_values.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(np.ravel(eigen_vectors[:, 0]))


def find_points_within_radius(source_cloud, target_cloud, radius):
    # Build KD-tree for target cloud
    tree = o3d.geometry.KDTreeFlann(target_cloud)

    # Get points from source cloud
    source_points = np.asarray(source_cloud.points)

    # Store indices of points within radius
    indices = []

    # Search for each point
    for i, point in enumerate(source_points):
        # Returns (num_neighbors, [indices], [distances²])
        [k, idx, _] = tree.search_radius_vector_3d(point, radius)
        if k > 0:
            indices.append(i)

    # Create new pointcloud with only the matching points
    matching_cloud = source_cloud.select_by_index(indices)
    return matching_cloud, indices


def decompose_rotation_matrix(R):
    """
    Decompose a 3x3 rotation matrix into Euler angles (in radians)
    Using Tait-Bryan angles with order XYZ
    """
    # Handle numerical errors that might make values slightly out of [-1,1]
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def clamp_quaternion_rotation_by_axis(quaternion, x_limit=X_LIMIT, y_limit=Y_LIMIT, z_limit=Z_LIMIT):
    """
    This function takes in a quaternion, and clamps the rotation per axis before returning as a constrained normalized quaternion"""
    norm_quaternion = quaternion/np.linalg.norm(quaternion)
    conversion_factor = np.pi/180
    x_radians, y_radians, z_radians = x_limit * conversion_factor, y_limit * \
        conversion_factor, z_limit * conversion_factor
    rot_matrix = Rotation.from_quat(
        norm_quaternion, scalar_first=True).as_matrix()
    axis_rotations = decompose_rotation_matrix(rot_matrix)
    constrained_angles = np.array([
        np.clip(axis_rotations[0], -x_radians, x_radians),
        np.clip(axis_rotations[1], -y_radians, y_radians),
        np.clip(axis_rotations[2], -z_radians, z_radians)
    ])
    cx, cy, cz = np.cos(constrained_angles)
    sx, sy, sz = np.sin(constrained_angles)

    # Build rotation matrix using XYZ order
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R_constrained = Rz @ Ry @ Rx
    r = Rotation.from_matrix(np.array(R_constrained))
    quaternion = r.as_quat(scalar_first=True)
    return quaternion/np.linalg.norm(quaternion), R_constrained


def generate_rotation_matrix(axis_rotations):
    """
    This function takes in a quaternion, and clamps the rotation per axis before returning as a constrained normalized quaternion"""

    cx, cy, cz = np.cos(axis_rotations)
    sx, sy, sz = np.sin(axis_rotations)

    # Build rotation matrix using XYZ order
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R_constrained = Rz @ Ry @ Rx
    r = Rotation.from_matrix(np.array(R_constrained))
    quaternion = r.as_quat(scalar_first=True)
    return quaternion/np.linalg.norm(quaternion), R_constrained


def find_closest_orthogonal(basis_vector, target_vector):
    """
    Find vector orthogonal to basis_vector that's closest to target_vector

    Args:
        basis_vector: First unit vector that defines primary axis
        target_vector: Second unit vector we want to get close to

    Returns:
        orthogonal unit vector closest to target_vector
    """
    # Project target onto basis
    projection = np.dot(target_vector, basis_vector) * basis_vector

    # Subtract projection to get orthogonal component
    orthogonal = target_vector - projection

    # Normalize result
    return orthogonal / np.linalg.norm(orthogonal)


def align_vectors_to_axes(vec_for_x, vec_for_y, vec_for_z):
    """
    Creates a rotation matrix that aligns three orthogonal vectors with the coordinate axes.

    Args:
        vec_for_x: 3D vector to be aligned with x-axis [1,0,0]
        vec_for_y: 3D vector to be aligned with y-axis [0,1,0]
        vec_for_z: 3D vector to be aligned with z-axis [0,0,1]

    Returns:
        rotation_matrix: 4x4 transformation matrix
    """

    # Create matrix from input vectors (each vector is a column)
    source_frame = np.column_stack([vec_for_x, vec_for_y, vec_for_z])

    # Create matrix for target coordinate frame
    target_frame = np.eye(3)  # Identity matrix represents standard basis

    # The rotation matrix is R = target_frame @ source_frame^T
    rotation_matrix = target_frame @ np.linalg.inv(source_frame)

    transform = np.eye(4)  # Create 4x4 identity matrix
    transform[:3, :3] = rotation_matrix

    return transform


def get_closest_indices(vertex_points, candidate_cloud):
    """
    vertex points is points you want to find matches to, candidate cloud is the cloud we're finding matches within.
    both are numpy arrays, of shape Nx3
    """
    matching_indices = []
    print("getting closest points")
    for point in vertex_points:
        norms = np.linalg.norm(candidate_cloud - point, axis=1)
        print(np.min(norms))
        matching_indices.append(
            np.argmin(norms))
    return np.array(matching_indices)


def get_new_mapping(vertex_mapping, original_vertices, new_vertices):
    new_dict = defaultdict(set)
    for (index, cluster) in tqdm(vertex_mapping.items()):
        norms = np.linalg.norm(new_vertices - original_vertices[index], axis=1)

        new_dict[cluster].add(np.argmin(norms))
    return new_dict


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
