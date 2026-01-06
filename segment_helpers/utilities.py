import os
import trimesh
import open3d as o3d


def load_meshes(directory_path, mesh_names):
    """
    Load meshes as trimesh objects from a directory.

    Args:
        directory_path: Path to the directory containing mesh files
        mesh_names: List of mesh names (without extension) to load

    Returns:
        List of trimesh.Trimesh objects in the same order as mesh_names
    """
    meshes = []

    # Common mesh file extensions to try
    extensions = ['.stl', '.obj', '.ply', '.off']

    for name in mesh_names:
        mesh = None
        for ext in extensions:
            filepath = os.path.join(directory_path, f"{name}{ext}")
            if os.path.exists(filepath):
                mesh = trimesh.load(filepath)
                break

        if mesh is None:
            raise FileNotFoundError(
                f"Could not find mesh file for '{name}' in {directory_path}")

        meshes.append(mesh)

    return meshes


def load_meshes_o3d(directory_path, mesh_names):
    """
    Load meshes as Open3D TriangleMesh objects from a directory.

    Args:
        directory_path: Path to the directory containing mesh files
        mesh_names: List of mesh names (without extension) to load

    Returns:
        List of open3d.geometry.TriangleMesh objects in the same order as mesh_names
    """
    meshes = []

    # Common mesh file extensions to try
    extensions = ['.stl', '.obj', '.ply', '.off']

    for name in mesh_names:
        mesh = None
        for ext in extensions:
            filepath = os.path.join(directory_path, f"{name}{ext}")
            if os.path.exists(filepath):
                mesh = o3d.io.read_triangle_mesh(filepath)
                mesh.compute_vertex_normals()
                break

        if mesh is None:
            raise FileNotFoundError(
                f"Could not find mesh file for '{name}' in {directory_path}")

        meshes.append(mesh)

    return meshes
