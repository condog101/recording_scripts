import open3d as o3d
import numpy as np
import cv2
from pyk4a import PyK4APlayback, ImageFormat, CalibrationType
import copy


def compute_mesh_vertex_distances(mesh1, mesh2):
    """
    Compare vertex distances between two meshes with corresponding vertices by index.

    Args:
        mesh1: First Open3D TriangleMesh
        mesh2: Second Open3D TriangleMesh (must have same number of vertices)

    Returns:
        dict: Statistics including mean, std, min, max, median distances
              and the per-vertex distance array
    """
    vertices1 = np.asarray(mesh1.vertices)
    vertices2 = np.asarray(mesh2.vertices)

    if len(vertices1) != len(vertices2):
        raise ValueError(
            f"Meshes have different vertex counts: {len(vertices1)} vs {len(vertices2)}")

    # Compute per-vertex Euclidean distances
    distances = np.linalg.norm(vertices1 - vertices2, axis=1)

    stats = {
        'num_vertices': len(distances),
        'mean': float(np.mean(distances)),
        'std': float(np.std(distances)),
        'min': float(np.min(distances)),
        'max': float(np.max(distances)),
        'median': float(np.median(distances)),
        'rmse': float(np.sqrt(np.mean(distances**2))),
        'percentile_95': float(np.percentile(distances, 95)),
        'percentile_99': float(np.percentile(distances, 99)),
        'distances': distances
    }

    return stats


def print_mesh_comparison(name1, name2, stats):
    """Print mesh comparison statistics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Mesh Vertex Distance Comparison")
    print(f"  Mesh 1: {name1}")
    print(f"  Mesh 2: {name2}")
    print(f"{'='*60}")
    print(f"  Vertices compared: {stats['num_vertices']}")
    print(f"  Mean distance:     {stats['mean']:.3f} mm")
    print(f"  Std deviation:     {stats['std']:.3f} mm")
    print(f"  Min distance:      {stats['min']:.3f} mm")
    print(f"  Max distance:      {stats['max']:.3f} mm")
    print(f"  Median distance:   {stats['median']:.3f} mm")
    print(f"  RMSE:              {stats['rmse']:.3f} mm")
    print(f"  95th percentile:   {stats['percentile_95']:.3f} mm")
    print(f"  99th percentile:   {stats['percentile_99']:.3f} mm")
    print(f"{'='*60}")


marker_to_depth_camera_path = "transform_ArucoBoard_to_depth_camera_9J4ophGG_20251209_162655.npy"
obj_path = "/home/connorscomputer/Desktop/imfusion_world_ct_12.obj"
mkv_path = "/home/connorscomputer/Desktop/9J4ophGG_20251209_162655.mkv"
board_path = "/home/connorscomputer/Desktop/hex30_fusion_coordinates_flipped.stl"

obj34_path = "/home/connorscomputer/Desktop/imfusion_world_ct_34.obj"
obj56_path = "/home/connorscomputer/Desktop/imfusion_world_ct_56.obj"
obj78_path = "/home/connorscomputer/Desktop/imfusion_world_ct_78.obj"


# obj12_path_flipped = "/home/connorscomputer/Desktop/imfusion_world_ct_12_flipped.obj"
# obj34_path_flipped = "/home/connorscomputer/Desktop/imfusion_world_ct_34_flipped.obj"
# obj56_path_flipped = "/home/connorscomputer/Desktop/imfusion_world_ct_56_flipped.obj"
# obj78_path_flipped = "/home/connorscomputer/Desktop/imfusion_world_ct_78_flipped.obj"

to_12_to_depth_camera_path = "transform_imfusion_world_ct_12_to_depth_camera.npy"
to_34_to_depth_camera_path = "transform_imfusion_world_ct_34_to_depth_camera.npy"
to_56_to_depth_camera_path = "transform_imfusion_world_ct_56_to_depth_camera_9J4ophGG_20251209_162655.npy"
to_78_to_depth_camera_path = "transform_imfusion_world_ct_78_to_depth_camera_9J4ophGG_20251209_162655.npy"

# Options
FLIP_POINTCLOUD_Y = False  # Set to True to negate Y coordinates of the point cloud


def main():
    # Load transformation matrices
    print("Loading transformation matrices...")
    y_flip = np.diag([1, -1, 1, 1])
    offset = 5.854 - (1.575)

    board_to_det = ([[-7.24840999e-01,  4.66900153e-02,  6.87332000e-01, 4.91281499e+01],
                     [6.88219001e-01,  4.22098675e-03,
                         7.25490000e-01, -3.94769367e+01],
                     [3.09710202e-02,  9.98900999e-01, -
                         3.51930009e-02, 382.078626 + offset],
                     [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

    board_to_det = np.array([[-7.24840999e-01,  4.66900153e-02,  6.87332000e-01, 4.51281499e+01],
                             [6.88219001e-01,  4.22098675e-03,
                                 7.25490000e-01, -4.14769367e+01],
                             [3.09710202e-02,  9.98900999e-01, -
                                 3.51930009e-02, 382.078626 + offset],
                             [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

    # this one is for marker 1-2
    world_to_cb_geom = np.array([[-0.193908782011579,  0.172948635328963,   0.96565426204032,  -41.9756705901941],  [-0.54440572870059, -0.837832310157343,  0.040736011272453,   23.4920032823171],  [0.816101578736842,  -0.51780864186904,  0.256617270636595,   375.991058989347],  [0,                  0,                  0,                  1]]

















                                )

    world_to_cb_geom_34 = np.array([[-0.263631606332836,  0.412890808711538,  0.871791004899627,  -40.5032604356369],  [-0.415939597940903, -0.864085880625352,  0.283460476554791,   4.96508905387752],  [0.87034052359242, -0.287883259300436,  0.399537985687756,   379.634495859459],  [0,                  0,                  0,                  1]]
                                   )

    world_to_cb_geom_56 = np.array([[-0.383993113405375,  0.439536394652021,  0.812008033601608,  -42.5473412481603],  [-0.684593119708782, -0.725647509665592, 0.0690503596190609,  -
                                                                                                                        58.0465628707011],  [0.619581753527848, -0.529380250380035,  0.579547238111725,   381.621274850155],  [0,                  0,                  0,                  1]])

    world_to_cb_geom_78 = np.array([[-0.333012399552806,  0.491726268380768,  0.804554546770079,  -42.9999616506563],  [-0.336740315298734, -0.859009323201799,  0.385627984857034,  -20.1916557846184],  [0.88074326667684, -0.142507051182432,  0.451644814612872,   379.496750809682],  [0,                  0,                  0,                  1]]

                                   )

    T_ct_tool = np.linalg.inv(board_to_det) @ world_to_cb_geom

    T_ct_tool_34 = np.linalg.inv(board_to_det) @ world_to_cb_geom_34

    T_ct_tool_56 = np.linalg.inv(board_to_det) @ world_to_cb_geom_56

    T_ct_tool_78 = np.linalg.inv(board_to_det) @ world_to_cb_geom_78

    T_depth_camera_marker = np.load(marker_to_depth_camera_path).astype(
        np.float64)  # Marker/Tool → Azure Kinect Depth Camera
    T_depth_camera_marker = T_depth_camera_marker @ y_flip
    print("\nT_ct_tool (CT → Tool):")
    print(T_ct_tool)
    print(f"Shape: {T_ct_tool.shape}")

    print("\nT_depth_camera_marker (Marker/Tool → Azure Kinect Depth Camera):")
    print(T_depth_camera_marker)
    print(f"Shape: {T_depth_camera_marker.shape}")

    # Load CT mesh
    print(f"\nLoading CT mesh from: {obj_path}")

    ### direct loaded transforms###
    ct_12_depth_camera_marker = np.load(to_12_to_depth_camera_path).astype(
        np.float64)

    ct_34_depth_camera_marker = np.load(to_34_to_depth_camera_path).astype(
        np.float64)

    ct_56_depth_camera_marker = np.load(to_56_to_depth_camera_path).astype(
        np.float64)

    ct_78_depth_camera_marker = np.load(to_78_to_depth_camera_path).astype(
        np.float64)

    ct_mesh = o3d.io.read_triangle_mesh(obj_path)
    ct_mesh.compute_vertex_normals()

    ct_mesh_copy = copy.deepcopy(ct_mesh)
    ct_mesh_copy.transform(ct_12_depth_camera_marker @ y_flip)

    ct_mesh_34 = o3d.io.read_triangle_mesh(obj34_path)
    ct_mesh_34.compute_vertex_normals()

    ct_mesh_34_copy = copy.deepcopy(ct_mesh_34)
    ct_mesh_34_copy.transform(ct_34_depth_camera_marker @ y_flip)

    ct_mesh_56 = o3d.io.read_triangle_mesh(obj56_path)
    ct_mesh_56.compute_vertex_normals()

    ct_mesh_56_copy = copy.deepcopy(ct_mesh_56)
    ct_mesh_56_copy.transform(ct_56_depth_camera_marker @ y_flip)

    ct_mesh_78 = o3d.io.read_triangle_mesh(obj78_path)
    ct_mesh_78.compute_vertex_normals()

    ct_mesh_78_copy = copy.deepcopy(ct_mesh_78)
    ct_mesh_78_copy.transform(ct_78_depth_camera_marker @ y_flip)

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

    combine = T_depth_camera_marker @ T_ct_tool

    combine_34 = T_depth_camera_marker @ T_ct_tool_34

    combine_56 = T_depth_camera_marker @ T_ct_tool_56

    combine_78 = T_depth_camera_marker @ T_ct_tool_78
    ct_mesh.transform(combine)
    ct_mesh_34.transform(combine_34)
    ct_mesh_56.transform(combine_56)

    ct_mesh_78.transform(combine_78)
    board_mesh.transform(T_depth_camera_marker)

    # Transform the CT frame too
    ct_origin_frame.transform(combine)

    mesh_stats_12 = compute_mesh_vertex_distances(ct_mesh, ct_mesh_copy)
    mesh_stats_34 = compute_mesh_vertex_distances(ct_mesh_34, ct_mesh_34_copy)
    mesh_stats_78 = compute_mesh_vertex_distances(ct_mesh_78, ct_mesh_78_copy)

    mesh_stats_56 = compute_mesh_vertex_distances(ct_mesh_56, ct_mesh_56_copy)

    print_mesh_comparison(
        "CT Mesh 12", "CT Mesh 12 (Direct Transform)", mesh_stats_12)
    print_mesh_comparison(
        "CT Mesh 34", "CT Mesh 34 (Direct Transform)", mesh_stats_34)
    print_mesh_comparison(
        "CT Mesh 78", "CT Mesh 78 (Direct Transform)", mesh_stats_78)

    print_mesh_comparison(
        "CT Mesh 56", "CT Mesh 56 (Direct Transform)", mesh_stats_56)

    # Color the CT mesh for visualization
    ct_mesh.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
    ct_mesh_34.paint_uniform_color([0.0, 1.0, 0.0])  # Green color

    ct_mesh_56.paint_uniform_color([0.0, 0.0, 1.0])  # Blue color

    ct_mesh_78.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow color

    ct_mesh_copy.paint_uniform_color([0.0, 1.0, 1.0])  # Red color
    ct_mesh_34_copy.paint_uniform_color([1.0, 0.0, 1.0])  # Green color

    ct_mesh_78_copy.paint_uniform_color([0.0, 0.0, 1.0])  # Yellow color

    # Compute distance between board_mesh and CT_mesh origins
    board_origin = np.array([0, 0, 0, 1])  # Origin in homogeneous coordinates
    ct_origin = np.array([0, 0, 0, 1])  # Origin in homogeneous coordinates

    # Transform origins to depth camera space
    board_origin_depth_camera = (T_depth_camera_marker @ board_origin)[:3]
    ct_origin_depth_camera = (combine @ ct_origin)[:3]

    # Compute distance
    distance = np.linalg.norm(
        board_origin_depth_camera - ct_origin_depth_camera)

    print(f"\n--- Origin Distance Analysis ---")
    print(
        f"Board mesh origin in depth camera space: [{board_origin_depth_camera[0]:.2f}, {board_origin_depth_camera[1]:.2f}, {board_origin_depth_camera[2]:.2f}] mm")
    print(
        f"CT mesh origin in depth camera space:    [{ct_origin_depth_camera[0]:.2f}, {ct_origin_depth_camera[1]:.2f}, {ct_origin_depth_camera[2]:.2f}] mm")
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

        # Get color image transformed to depth camera space
        transformed_color = capture.transformed_color
        if transformed_color is not None:
            transformed_color_bgr = transformed_color[..., (2, 1, 0)]
        else:
            print("Warning: Could not get transformed color, using gray")
            transformed_color_bgr = None

        # Get point cloud from depth (native depth camera space)
        points = capture.depth_point_cloud.reshape(
            (-1, 3)).astype('float64')
        if transformed_color_bgr is not None:
            colors = transformed_color_bgr.reshape((-1, 3))
        else:
            colors = np.full((points.shape[0], 3), 128, dtype=np.uint8)

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
        print("Red mesh: CT scan transformed to depth camera space")
        print("Colored points: Azure Kinect depth point cloud")
        print("Large RGB axes (100mm): Azure Kinect depth camera coordinate frame")
        print("Medium RGB axes (50mm): CT mesh coordinate frame")
        print("Small RGB axes (30mm): Board/ArUco marker coordinate frame")

        # Add coordinate frames to the visualization
        o3d.visualization.draw_geometries(
            [pcd, ct_mesh_56, ct_mesh_78, ct_mesh_78, ct_mesh_78_copy, camera_frame,
             board_mesh],
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
