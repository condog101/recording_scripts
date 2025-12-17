from argparse import ArgumentParser
import cv2
import numpy as np
from typing import Optional, Tuple
from pyk4a import PyK4APlayback, ImageFormat, CalibrationType
import json
import matplotlib.pyplot as plt
import open3d as o3d


def load_intrinsics_from_calibration(calibration, camera_type=CalibrationType.DEPTH):
    """Extract intrinsics from PyK4A calibration."""
    K = calibration.get_camera_matrix(camera_type)
    dist = calibration.get_distortion_coefficients(camera_type)
    return K.astype(np.float32), dist.astype(np.float32)


def get_color_to_depth_transform(calibration):
    """
    Get the 4x4 transformation matrix from color camera space to depth camera space.
    """
    rotation, translation = calibration.get_extrinsic_parameters(
        CalibrationType.COLOR, CalibrationType.DEPTH
    )
    # translation is returned in meters, convert to mm for consistency
    translation_mm = translation.flatten() * 1000.0

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = translation_mm
    return T


def rvec_tvec_to_T(rvec, tvec):
    """Convert rotation vector and translation to 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


class BoardTracker:
    def __init__(self, K: np.ndarray, dist: np.ndarray, marker_size_mm=20.0, aruco_dict_name="DICT_6X6_50"):
        self.K = K.astype(np.float32)
        self.dist = dist.astype(np.float32)
        self.marker_size_mm = marker_size_mm
        self.marker_stl = "/home/connorscomputer/Desktop/imfusion_world_ct_34_flipped.stl"

        dict_map = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        }
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            dict_map.get(aruco_dict_name, cv2.aruco.DICT_6X6_50))
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.detector_params)

        self.tools = {}
        self._last_ids = None
        self._last_corners = None

    def initialize_from_config(self, board_config_json_path: str):
        with open(board_config_json_path, "r") as f:
            cfg = json.load(f)

        self.tools.clear()
        for tool in cfg.get("toolList", []):
            name = tool["id"]
            ids = tool["marker_ids"]
            corners = tool["marker_corners_m"]
            per_marker = {}
            for mid, m_corners in zip(ids, corners):
                # Keep in mm to match Azure Kinect point cloud units
                arr = np.array(m_corners, dtype=np.float32)
                per_marker[int(mid)] = arr
            self.tools[name] = per_marker

    def detect(self, image: np.ndarray):
        corners, ids, _ = self.detector.detectMarkers(image)
        self._last_corners = corners
        self._last_ids = ids
        return corners, ids

    def _collect_tool_correspondences(self, tool_name):
        marker_map = self.tools[tool_name]
        if self._last_ids is None or len(self._last_ids) == 0:
            return None, None

        id_to_idx = {int(_id): idx for idx, _id in enumerate(
            self._last_ids.flatten())}
        obj_pts, img_pts = [], []

        for mid, obj_corners in marker_map.items():
            if mid in id_to_idx:
                di = id_to_idx[mid]
                img_corners = self._last_corners[di].reshape(-1, 2)
                obj_pts.append(obj_corners)
                img_pts.append(img_corners)

        if not obj_pts:
            return None, None

        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)
        return obj_pts, img_pts

    def RequestPose(self, tool_name: str):
        if tool_name not in self.tools:
            return False, np.eye(4, dtype=np.float32), None, None, {}

        obj_pts, img_pts = self._collect_tool_correspondences(tool_name)
        if obj_pts is None:
            return False, np.eye(4, dtype=np.float32), None, None, {}

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=self.K,
            distCoeffs=self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0,
            confidence=0.99,
            iterationsCount=100
        )
        if not success:
            return False, np.eye(4, dtype=np.float32), None, None, {}

        # Compute projection error metrics
        projected_pts, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, self.K, self.dist)
        projected_pts = projected_pts.reshape(-1, 2)

        # Calculate per-point errors
        errors = np.linalg.norm(img_pts - projected_pts, axis=1)

        metrics = {
            'mean_error': float(np.mean(errors)),
            'max_error': float(np.max(errors)),
            'min_error': float(np.min(errors)),
            'std_error': float(np.std(errors)),
            'median_error': float(np.median(errors)),
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'num_points': len(obj_pts),
            'num_inliers': int(len(inliers)) if inliers is not None else len(obj_pts),
            'inlier_ratio': float(len(inliers) / len(obj_pts)) if inliers is not None else 1.0
        }

        return True, rvec_tvec_to_T(rvec, tvec), rvec, tvec, metrics

    def GetTrackableNames(self):
        return list(self.tools.keys())

    def annotate(self, image: np.ndarray):
        """Draw detected markers and coordinate axes on image."""
        out = image.copy()

        # Draw detected markers with IDs
        if self._last_ids is not None and len(self._last_ids) > 0:
            cv2.aruco.drawDetectedMarkers(
                out, self._last_corners, self._last_ids)

        # Draw coordinate frame for each detected tool
        for name in self.GetTrackableNames():
            found, T, rvec, tvec, metrics = self.RequestPose(name)

            if found:
                # Draw 3D coordinate axes (length = marker size)
                cv2.drawFrameAxes(out, self.K, self.dist, rvec, tvec,
                                  self.marker_size_mm, 3)

                # Add text label with tool name
                origin_2d, _ = cv2.projectPoints(
                    np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                    rvec, tvec, self.K, self.dist
                )
                origin_2d = tuple(origin_2d[0, 0].astype(int))
                cv2.putText(out, name, (origin_2d[0] + 10, origin_2d[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display position
                pos_text = f"Pos: ({T[0,3]:.1f}, {T[1,3]:.1f}, {T[2,3]:.1f}) mm"
                cv2.putText(out, pos_text, (origin_2d[0] + 10, origin_2d[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Display projection error
                error_text = f"RMSE: {metrics['rmse']:.2f}px ({metrics['num_inliers']}/{metrics['num_points']} pts)"
                cv2.putText(out, error_text, (origin_2d[0] + 10, origin_2d[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)

        return out


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


def main() -> None:

    filename = "/home/connorscomputer/Desktop/9J4ophGG_20251209_162542.mkv"
    spine_config = "/home/connorscomputer/recording_scripts/imfusion_world_ct_34_marker_config.json"
    offset = 0.0

    playback = PyK4APlayback(filename)
    playback.open()

    info(playback)

    # Get intrinsics from first capture
    capture = playback.get_next_capture()
    K_color, dist_color = load_intrinsics_from_calibration(
        capture._calibration, CalibrationType.COLOR)

    # Get the extrinsic transform from color camera to depth camera
    T_color_to_depth = get_color_to_depth_transform(capture._calibration)
    print(f"\nColor to Depth extrinsic transform:")
    print(
        f"  Translation (mm): [{T_color_to_depth[0,3]:.2f}, {T_color_to_depth[1,3]:.2f}, {T_color_to_depth[2,3]:.2f}]")

    # Seek to offset if specified
    if offset != 0.0:
        playback.seek(int(offset * 1000000))
        capture = playback.get_next_capture()

    # Initialize tracker with COLOR camera intrinsics for detection and pose estimation
    # We'll chain the transform afterwards to get depth camera space
    tracker = BoardTracker(K_color, dist_color, marker_size_mm=20.0,
                           aruco_dict_name="DICT_4X4_50")
    tracker.initialize_from_config(spine_config)

    print(f"\nTracking tools: {tracker.GetTrackableNames()}")
    print(f"Using 6x6 ArUco markers, 30mm size")
    print(f"Detection in COLOR camera (high-res), transform chained to DEPTH camera space")

    # Process one frame
    if capture.color is not None and capture.depth is not None:
        # Prepare color image for ArUco detection
        capture._color = cv2.cvtColor(cv2.imdecode(
            capture.color, cv2.IMREAD_COLOR), cv2.COLOR_BGR2BGRA)
        capture._color_format = ImageFormat.COLOR_BGRA32

        color_bgr = capture._color[..., (2, 1, 0)]

        # Get color image transformed to depth camera space (for visualization only)
        transformed_color = capture.transformed_color
        if transformed_color is not None:
            transformed_color_bgr = transformed_color[..., (2, 1, 0)]
        else:
            print("Warning: Could not get transformed color image")
            transformed_color_bgr = None

        # Detect markers in the HIGH-RES color image (better detection)
        corners, ids = tracker.detect(color_bgr)

        print(f"\nDetected {len(ids) if ids is not None else 0} markers")
        if ids is not None:
            print(f"Marker IDs: {ids.flatten()}")

        # Annotate the color image
        annotated = tracker.annotate(color_bgr)

        # Check pose for each tool
        for name in tracker.GetTrackableNames():
            found, T_marker_to_color, rvec, tvec, metrics = tracker.RequestPose(
                name)
            if found:
                # Chain transforms: marker -> color camera -> depth camera
                # T_marker_to_depth = T_color_to_depth @ T_marker_to_color
                T_marker_to_depth = T_color_to_depth @ T_marker_to_color.astype(
                    np.float64)

                print(f"\n{name} pose found:")
                print(
                    f"  Position (color camera space): [{T_marker_to_color[0,3]:.1f}, {T_marker_to_color[1,3]:.1f}, {T_marker_to_color[2,3]:.1f}] mm")
                print(
                    f"  Position (depth camera space): [{T_marker_to_depth[0,3]:.1f}, {T_marker_to_depth[1,3]:.1f}, {T_marker_to_depth[2,3]:.1f}] mm")
                print(f"\n  Projection Error Metrics:")
                print(f"    RMSE:          {metrics['rmse']:.3f} pixels")
                print(f"    Mean Error:    {metrics['mean_error']:.3f} pixels")
                print(
                    f"    Median Error:  {metrics['median_error']:.3f} pixels")
                print(f"    Max Error:     {metrics['max_error']:.3f} pixels")
                print(f"    Min Error:     {metrics['min_error']:.3f} pixels")
                print(f"    Std Dev:       {metrics['std_error']:.3f} pixels")
                print(
                    f"    Points Used:   {metrics['num_inliers']}/{metrics['num_points']} (inlier ratio: {metrics['inlier_ratio']:.2%})")

                # Save transform as .npy file (in depth camera space)
                np.save(
                    f"transform_{name}_to_depth_camera.npy", T_marker_to_depth)
                print(
                    f"\n  Saved transform for {name} to transform_{name}_to_depth_camera.npy")
            else:
                print(f"\n{name}: Not detected")

        # Display using matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title("ArUco Marker Detection - Single Frame")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        # Get depth point cloud for 3D visualization
        if capture.depth is not None:
            print("\nGenerating 3D visualization with point cloud and mesh overlay...")

            # Get point cloud from depth (native depth camera space)
            points = capture.depth_point_cloud.reshape(
                (-1, 3)).astype('float64')
            # Use transformed color for point cloud coloring (aligned to depth)
            colors_pc = transformed_color_bgr.reshape(
                (-1, 3)) if transformed_color_bgr is not None else np.full_like(points, 128)

            # Filter out zero/invalid points
            valid_mask = (points[:, 2] > 0)
            points = points[valid_mask]
            colors_pc = colors_pc[valid_mask]

            # Azure Kinect point cloud coordinate system:
            # X: right, Y: down, Z: forward (away from camera)
            # This matches OpenCV/ArUco convention, so no flip should be needed
            # However, if there's a mismatch, try uncommenting one of these:
            # Option 1: Flip Y axis
            # points[:, 1] = -points[:, 1]
            # Option 2: Flip X and Y axes
            # points[:, 0] = -points[:, 0]
            # points[:, 1] = -points[:, 1]

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(
                (colors_pc / 255).astype('float64'))
            print(f"Point cloud has {len(pcd.points)} points")

            # Prepare geometries for visualization
            geometries = [pcd]

            # Add camera coordinate frame at origin
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=100.0, origin=[0, 0, 0]
            )
            geometries.append(camera_frame)

            # Load and transform marker mesh for each detected tool
            for name in tracker.GetTrackableNames():
                found, T_marker_to_color, rvec, tvec, metrics = tracker.RequestPose(
                    name)
                if found:
                    # Chain transforms to get depth camera space
                    T_marker_to_depth = T_color_to_depth @ T_marker_to_color.astype(
                        np.float64)

                    print(f"\nLoading marker mesh: {tracker.marker_stl}")
                    marker_mesh = o3d.io.read_triangle_mesh(tracker.marker_stl)

                    if marker_mesh.has_vertices():
                        marker_mesh.compute_vertex_normals()

                        # Debug: Print transform details
                        print(
                            f"\n  Transform matrix for {name} (depth camera space):")
                        print(
                            f"  Translation (mm): [{T_marker_to_depth[0,3]:.1f}, {T_marker_to_depth[1,3]:.1f}, {T_marker_to_depth[2,3]:.1f}]")

                        # Get mesh center before transform
                        mesh_center_before = marker_mesh.get_center()
                        print(
                            f"  Mesh center before transform: {mesh_center_before}")

                        # Apply the transform to place mesh in depth camera coordinates
                        marker_mesh.transform(T_marker_to_depth)

                        # Get mesh center after transform
                        mesh_center_after = marker_mesh.get_center()
                        print(
                            f"  Mesh center after transform: {mesh_center_after}")

                        # Find nearest point cloud points to mesh center for comparison
                        distances = np.linalg.norm(np.asarray(
                            pcd.points) - mesh_center_after, axis=1)
                        nearest_idx = np.argmin(distances)
                        nearest_dist = distances[nearest_idx]
                        print(
                            f"  Distance from mesh center to nearest point cloud point: {nearest_dist:.1f} mm")

                        # Color the mesh for visibility
                        marker_mesh.paint_uniform_color(
                            [0.0, 1.0, 0.0])  # Green

                        geometries.append(marker_mesh)

                        # Add coordinate frame at the marker origin (transformed to depth space)
                        marker_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=50.0, origin=[0, 0, 0]
                        )
                        marker_frame.transform(T_marker_to_depth)
                        geometries.append(marker_frame)

                        print(
                            f"  Mesh transformed to depth camera space for tool: {name}")
                    else:
                        print(
                            f"  Warning: Could not load mesh from {tracker.marker_stl}")

            # Visualize
            print("\n" + "="*60)
            print("3D Visualization")
            print("="*60)
            print("Green mesh: Marker STL transformed to depth camera space")
            print("Colored points: Azure Kinect point cloud")
            print("Large RGB axes (100mm): Camera coordinate frame")
            print("Medium RGB axes (50mm): Marker coordinate frame")
            print("\nControls:")
            print("  - Mouse: Rotate view")
            print("  - Scroll: Zoom in/out")
            print("  - Ctrl + Mouse: Pan")
            print("  - Q or ESC: Close window")

            o3d.visualization.draw_geometries(
                geometries,
                window_name="ArUco Tracking - Point Cloud with Mesh Overlay",
                width=1280,
                height=720
            )

    playback.close()


if __name__ == "__main__":
    main()
