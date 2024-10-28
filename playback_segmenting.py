from argparse import ArgumentParser
import cv2
import numpy as np
from pyk4a import PyK4APlayback, ImageFormat
import open3d as o3d
import pickle
from typing import Optional
from utils import info
from markup_sequence import record_pointed_spots, get_3d_coordinates, raw_get_3d_coordinates, play_single_initial_frame_mark_both
from registration import rough_register_via_correspondences, source_icp_transform
from armature_utils import get_joint_positions
import bezier
from collections import defaultdict


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


def subsample_mesh(colors, points, bbox_params, calibration, depth_map, offset=10.0):
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

    combined_mask = mask_x & mask_y

    return colors[combined_mask], points[combined_mask]


def partition(mesh):
    color_map = defaultdict(list)
    colors = np.asarray(mesh.vertex_colors)
    for ind in range(len(colors)):
        r, g, b = colors[ind]
        color_map[(r, g, b)].append(ind)
    print(len(color_map))
    return color_map


class BallJoint:
    def __init__(self, position):
        self.position = np.array(position)
        # Initialize quaternion to identity rotation [1,0,0,0]
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    def set_rotation_axis_angle(self, axis, angle):
        """Set joint rotation using axis-angle representation"""
        axis = axis / np.linalg.norm(axis)  # normalize axis
        half_angle = angle / 2.0
        self.quaternion[0] = np.cos(half_angle)
        self.quaternion[1:] = axis * np.sin(half_angle)

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def rotate_point(self, point):
        """Apply rotation to a point using quaternion rotation"""
        # Convert point to quaternion form (w=0)
        p_quat = np.array([0.0, *point])

        # Get quaternion conjugate
        q_conj = np.array([self.quaternion[0], *(-self.quaternion[1:])])

        # Rotate point: q * p * q'
        rotated = self.quaternion_multiply(
            self.quaternion,
            self.quaternion_multiply(p_quat, q_conj)
        )

        # Return rotated point (ignore w component)
        return rotated[1:]

    def transform_point(self, point):
        """Apply full transformation (rotation + translation) to point"""
        # First rotate
        rotated_point = self.rotate_point(point)
        # Then translate
        return rotated_point + self.position

    def to_matrix(self):
        """Convert to 4x4 transformation matrix"""
        w, x, y, z = self.quaternion

        matrix = np.array([
            [1-2*y*y-2*z*z,  2*x*y-2*w*z,    2*x*z+2*w*y,    self.position[0]],
            [2*x*y+2*w*z,    1-2*x*x-2*z*z,  2*y*z-2*w*x,    self.position[1]],
            [2*x*z-2*w*y,    2*y*z+2*w*x,    1-2*x*x-2*y*y,  self.position[2]],
            [0,              0,              0,              1]
        ])

        return matrix


def extract_control_points(spine_points: np.array, control_points: int):

    remaining_points = spine_points[1:-1]
    indices = np.linspace(0, len(remaining_points)-1,
                          control_points-2, dtype=int)
    return spine_points[[0, *indices, -1]]


def play(playback: PyK4APlayback, offset: float, record_spots: bool, file_short: str, mesh_filepath: Optional[str] = None,
         baseline_frame: int = 0, record_mesh_points: bool = False, scale_scene=10.0, record_spline_points=False):

    if record_spots:
        record_pointed_spots(playback, file_short)

    coordinates_file = f"stored_coordinates_{file_short}.pkl"
    time_file = f"times_{file_short}.pkl"

    with open(coordinates_file, 'rb') as f:
        coordinates = pickle.load(f)

    with open(time_file, 'rb') as f:
        times = pickle.load(f)

    spine_mesh = o3d.io.read_triangle_mesh(mesh_filepath)
    if spine_mesh is not None:
        if record_mesh_points:
            pp_list = play_single_initial_frame_mark_both(
                spine_mesh, playback, offset, baseline_frame)

            print(pp_list)
            with open(f"meshpoints_{file_short}.npy", 'wb+') as f:
                # source, destination
                np.save(f, pp_list)
        else:
            with open(f"meshpoints_{file_short}.npy", 'rb') as f:
                # source, destination
                pp_list = np.load(f)

        if record_spline_points:
            spline_list = play_single_initial_frame_mark_both(
                spine_mesh, playback, offset, baseline_frame)

            print(spline_list)
            with open(f"spline_points_{file_short}.npy", 'wb+') as f:
                # source, destination
                np.save(f, spline_list)
        else:
            with open(f"spline_points_{file_short}.npy", 'rb') as f:
                # source, destination
                spline_list = np.load(f)

    # pplist, for array is source, second is target

    spine_points_spline = spline_list[0]
    surg_points_spline = spline_list[1]

    # number of joints is number of vertebrae -1
    # number of bezier control points = number of joints + 2

    partition_map = partition(spine_mesh)

    links = get_joint_positions(partition_map, spine_mesh)
    print(f"{links - 1} joints fitted")
    joint_positions = [x[1] for x in links[:-1]]
    num_joints = len(partition_map) - 1

    if len(spine_points_spline) != surg_points_spline:
        raise ValueError(
            "Curve points on ref frame and mesh are different lengths")

    control_point_count = num_joints + 2

    select_spine_points_spline = extract_control_points(
        spine_points_spline, control_point_count)
    select_surg_points_spline = extract_control_points(
        surg_points_spline, control_point_count)

    split_spine_points_spline = select_spine_points_spline.T
    split_surg_points_spline = select_surg_points_spline.T

    curve_spine = bezier.Curve(
        split_spine_points_spline, degree=control_point_count-1)
    curve_surg = bezier.Curve(split_surg_points_spline,
                              degree=control_point_count-1)

    # fit bezier curve to both spine process approximations
    # get link between joints of vertebrae, and control points of vertebrae
    # find parameter of joints to minimize difference between joints of

    geometry = o3d.geometry.PointCloud()

    frame_index = 0

    first = True

    pcd_spine = o3d.geometry.PointCloud()
    pcd_spine.points = spine_mesh.vertices
    pcd_spine.normals = spine_mesh.vertex_normals

    source_whole_rough = o3d.geometry.PointCloud()
    source_whole_rough.points = o3d.utility.Vector3dVector(pp_list[0])
    source_whole_rough.colors = o3d.utility.Vector3dVector(
        np.array([[0, 1, 0] for _ in pp_list[1]]))

    source_scene_rough = o3d.geometry.PointCloud()
    source_scene_rough.points = o3d.utility.Vector3dVector(
        pp_list[1] * scale_scene)
    source_scene_rough.colors = o3d.utility.Vector3dVector(
        np.array([[1, 0, 0] for _ in pp_list[1]]))

    rough_transform = rough_register_via_correspondences(
        source_whole_rough, source_scene_rough)

    # do registration for 1st

    visualize = True

    if visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

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
                    subsample_colors, subsample_points = subsample_mesh(
                        colors, points, ((x_c, y_c), w_c, h_c), capture._calibration, capture.transformed_depth, offset=20.0)

                    geometry.points = o3d.utility.Vector3dVector(
                        subsample_points * scale_scene)
                    geometry.colors = o3d.utility.Vector3dVector(
                        (subsample_colors/255).astype('float64'))

                    icp_transform = source_icp_transform(
                        pcd_spine, geometry, rough_transform, threshold=10)

                    inv_fine_transform = np.linalg.inv(icp_transform)

                    spine_mesh.transform(icp_transform)
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
                    spine_mesh.transform(inv_fine_transform)

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
