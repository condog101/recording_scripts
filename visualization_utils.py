import open3d as o3d
import numpy as np


# default red color
def create_arrow(start_point, vector, scale=1.0, color=[1, 0, 0]):
    """
    Create an arrow from a start point along a vector

    Args:
        start_point (np.array): Starting point of arrow [x, y, z]
        vector (np.array): Direction vector [dx, dy, dz]
        scale (float): Scale factor for arrow size
        color (list): RGB color for the arrow [r, g, b]

    Returns:
        o3d.geometry.TriangleMesh: Arrow mesh
    """
    # Normalize the vector
    vector = np.array(vector, dtype=np.float64)
    length = np.linalg.norm(vector)

    if length == 0:
        raise ValueError("Vector length cannot be zero")

    vector = vector / length

    # Create arrow geometry
    cylinder_radius = 0.1 * scale
    cone_radius = 0.2 * scale
    cylinder_height = 0.8 * scale
    cone_height = 0.2 * scale

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height
    )

    # Compute rotation to align with vector
    # Default arrow points in positive y direction
    default_direction = np.array([0, 0, 1])

    # Compute rotation axis and angle
    rotation_axis = np.cross(default_direction, vector)
    rotation_axis_length = np.linalg.norm(rotation_axis)

    if rotation_axis_length > 0:  # If vectors aren't parallel
        rotation_axis = rotation_axis / rotation_axis_length
        angle = np.arccos(np.dot(default_direction, vector))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis * angle)
        arrow.rotate(R, center=(0, 0, 0))

    # # Scale the arrow by the vector length
    # arrow.scale(length, center=[0, 0, 0])

    # Translate to start point
    arrow.translate(start_point)

    # Color the arrow
    arrow.paint_uniform_color(color)

    return arrow
