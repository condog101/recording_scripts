import open3d as o3d
import numpy as np
import cv2
from pyk4a import PyK4APlayback, ImageFormat, CalibrationType

ct_to_tool_path = "/home/connorscomputer/Desktop/AnimalXRays/OldSoloOriginal/ct_to_tool_transform.npy"
marker_to_rgbd_path = "transform_ArucoBoard_to_camera.npy"
obj_path = "/home/connorscomputer/Desktop/ProcessedOldSpine/RedoneSegments/Segmentation_1.obj"
mkv_path = "/home/connorscomputer/Downloads/old2.mkv"
board_path = "/home/connorscomputer/Downloads/arucoblender.obj"


def enforce_no_scaling(T):
    R = T[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_no_scale = U @ Vt
    T[:3, :3] = R_no_scale
    return T


def main():
    # Load transformation matrices
    print("Loading transformation matrices...")
    T_ct_tool = np.load(ct_to_tool_path).astype(np.float64)  # CT → Tool
    T_camera_marker = np.load(marker_to_rgbd_path).astype(
        np.float64)  # Marker/Tool → Azure Kinect Camera
    T_camera_marker[:3, 3] = T_camera_marker[:3, 3] * 1000.0
    print("\nT_ct_tool (CT → Tool):")
    print(T_ct_tool)
    print(f"Shape: {T_ct_tool.shape}")

    print("\nT_camera_marker (Marker/Tool → Azure Kinect Camera):")
    print(T_camera_marker)
    print(f"Shape: {T_camera_marker.shape}")

    # Load CT mesh
    print(f"\nLoading CT mesh from: {obj_path}")
    board_mesh = o3d.io.read_triangle_mesh(board_path)
    ct_mesh = o3d.io.read_triangle_mesh(obj_path)
    ct_mesh.compute_vertex_normals()

    board_mesh.compute_vertex_normals()
    flip = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    # board_mesh.transform(flip)
    board_mesh.transform(T_camera_marker @ flip)

    # Flip normals if they're reversed
    print("Flipping normals...")
    ct_mesh.triangle_normals = o3d.utility.Vector3dVector(
        -np.asarray(ct_mesh.triangle_normals))
    ct_mesh.vertex_normals = o3d.utility.Vector3dVector(
        -np.asarray(ct_mesh.vertex_normals))

    print(
        f"CT mesh has {len(ct_mesh.vertices)} vertices and {len(ct_mesh.triangles)} triangles")

    print("Applying transformation to CT mesh...")
    combine = enforce_no_scaling(T_camera_marker @ T_ct_tool)

    ct_mesh.transform(combine)

    # Flip Y and Z axes
    # print("Flipping Y and Z axes for mesh...")

    # Color the CT mesh for visualization
    ct_mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Red color

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

        # Visualize
        print("\nVisualizing transforms...")
        print("Red mesh: CT scan transformed to camera space")
        print("Colored points: Azure Kinect point cloud")
        print("RGB axes: Azure Kinect camera coordinate frame")

        print("Rotating mesh around Z axis by 180 degrees...")

        o3d.visualization.draw_geometries(
            [pcd, ct_mesh, camera_frame, board_mesh],
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
