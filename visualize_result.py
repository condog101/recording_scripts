import open3d as o3d
import numpy as np
import cv2
from pyk4a import PyK4APlayback, ImageFormat, CalibrationType


marker_to_rgbd_path = "transform_ArucoBoard_to_camera.npy"
obj_path = "/home/connorscomputer/Desktop/imfusion_world_ct_12.obj"
mkv_path = "/home/connorscomputer/Desktop/9J4ophGG_20251209_162542.mkv"
board_path = "/home/connorscomputer/Desktop/hex30_fusion_coordinates_flipped.stl"

# Options
FLIP_POINTCLOUD_Y = False  # Set to True to negate Y coordinates of the point cloud


def main():
    # Load transformation matrices
    print("Loading transformation matrices...")
    y_flip = np.diag([1, -1, 1, 1])
    offset = 382.078626 + 6.854 + (1.575)
    board_to_det = np.array([[-7.24840999e-01,  4.66900153e-02,  6.87332000e-01, 4.91281499e+01],
                             [6.88219001e-01,  4.22098675e-03,
                                 7.25490000e-01, -4.04769367e+01],
                             [3.09710202e-02,  9.98900999e-01, -
                                 3.51930009e-02, offset],
                             [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]],
                            dtype=np.float32)

    # this one is for marker 1-2
    world_to_cb_geom = np.array([[-2.41840158e-01,  2.16010175e-01,  9.45966671e-01,
                                  -3.80291390e+01],
                                 [-5.21356971e-01, -8.51150466e-01,  6.10720340e-02,
                                  2.86327744e+01],
                                 [8.18352154e-01, -4.78416648e-01,  3.18460773e-01,
                                  3.77418145e+02],
                                 [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                  1.00000000e+00]])
    T_ct_tool = np.linalg.inv(board_to_det) @ world_to_cb_geom

    T_camera_marker = np.load(marker_to_rgbd_path).astype(
        np.float64)  # Marker/Tool → Azure Kinect Camera
    T_camera_marker = T_camera_marker @ y_flip
    print("\nT_ct_tool (CT → Tool):")
    print(T_ct_tool)
    print(f"Shape: {T_ct_tool.shape}")

    print("\nT_camera_marker (Marker/Tool → Azure Kinect Camera):")
    print(T_camera_marker)
    print(f"Shape: {T_camera_marker.shape}")

    # Load CT mesh
    print(f"\nLoading CT mesh from: {obj_path}")

    ct_mesh = o3d.io.read_triangle_mesh(obj_path)
    ct_mesh.compute_vertex_normals()

    board_mesh = o3d.io.read_triangle_mesh(board_path)
    board_mesh.compute_vertex_normals()

    # Create coordinate frame for CT mesh origin (before transformation)
    ct_origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=50.0, origin=[0, 0, 0])
    # Keep default RGB colors (Red=X, Green=Y, Blue=Z)
    ct_origin_frame.compute_vertex_normals()

    # Flip normals if they're reversed
    print("Flipping normals...")
    ct_mesh.triangle_normals = o3d.utility.Vector3dVector(
        -np.asarray(ct_mesh.triangle_normals))
    ct_mesh.vertex_normals = o3d.utility.Vector3dVector(
        -np.asarray(ct_mesh.vertex_normals))

    print(
        f"CT mesh has {len(ct_mesh.vertices)} vertices and {len(ct_mesh.triangles)} triangles")

    print("Applying transformation to CT mesh...")
    # Apply same transformation to the CT coordinate frame

    combine = T_camera_marker @ T_ct_tool
    ct_mesh.transform(combine)
    board_mesh.transform(T_camera_marker)

    # Transform the CT frame too
    ct_origin_frame.transform(combine)

    # Color the CT mesh for visualization
    ct_mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Red color

    # Compute distance between board_mesh and CT_mesh origins
    board_origin = np.array([0, 0, 0, 1])  # Origin in homogeneous coordinates
    ct_origin = np.array([0, 0, 0, 1])  # Origin in homogeneous coordinates

    # Transform origins to camera space
    board_origin_camera = (T_camera_marker @ board_origin)[:3]
    ct_origin_camera = (combine @ ct_origin)[:3]

    # Compute distance
    distance = np.linalg.norm(board_origin_camera - ct_origin_camera)

    print(f"\n--- Origin Distance Analysis ---")
    print(
        f"Board mesh origin in camera space: [{board_origin_camera[0]:.2f}, {board_origin_camera[1]:.2f}, {board_origin_camera[2]:.2f}] mm")
    print(
        f"CT mesh origin in camera space:    [{ct_origin_camera[0]:.2f}, {ct_origin_camera[1]:.2f}, {ct_origin_camera[2]:.2f}] mm")
    print(
        f"Distance between origins: {distance:.2f} mm ({distance/10:.2f} cm)")

    # Load Azure Kinect video
    print(f"\nLoading Azure Kinect video: {mkv_path}")
    playback = PyK4APlayback(mkv_path)
    playback.open()
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")

    # Get one frame
    capture = playback.get_next_capture()

    if capture.color is not None and capture.depth is not None:
        print("Processing frame...")

        # Prepare color image
        capture._color = cv2.cvtColor(cv2.imdecode(
            capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
        capture._color_format = ImageFormat.COLOR_BGRA32
        color_bgr = capture._color[..., (2, 1, 0)]

        # Get point cloud from depth
        points = capture.transformed_depth_point_cloud.reshape(
            (-1, 3)).astype('float64')
        colors = color_bgr.reshape((-1, 3))

        # Filter out zero points
        valid_mask = (points[:, 2] > 0)
        points = points[valid_mask]
        colors = colors[valid_mask]

        # Flip Y coordinates if enabled
        if FLIP_POINTCLOUD_Y:
            print("Flipping Y coordinates of point cloud (negating Y)...")
            points[:, 1] = -points[:, 1]

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(
            (colors / 255).astype('float64'))
        print(f"Point cloud has {len(pcd.points)} points")

        # Flip X and Y coordinates of the point cloud
        print("Flipping X and Y coordinates for point cloud...")

        # Apply same transform to coordinate frame to keep it aligned
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=100.0, origin=[0, 0, 0])
        # Ensure colors are preserved (Red=X, Green=Y, Blue=Z)
        camera_frame.compute_vertex_normals()

        # Visualize
        print("\nVisualizing transforms...")
        print("Red mesh: CT scan transformed to camera space")
        print("Colored points: Azure Kinect point cloud")
        print("Large RGB axes (100mm): Azure Kinect camera coordinate frame")
        print("Medium RGB axes (50mm): CT mesh coordinate frame")
        print("Small RGB axes (30mm): Board/ArUco marker coordinate frame")

        # Add coordinate frames to the visualization
        o3d.visualization.draw_geometries(
            [pcd, ct_mesh, camera_frame,
                ct_origin_frame, board_mesh],
            window_name="CT in Azure Kinect Space",
            width=1280,
            height=720,
            left=50,
            top=50
        )
    else:
        print("Error: Could not read color or depth from capture")

    playback.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
