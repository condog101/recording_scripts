import numpy as np
import trimesh
from skimage import measure
import SimpleITK as sitk
import nibabel as nib
import open3d as o3d


def mesh_to_ct(ply_path, output_path, voxel_size=0.25, hu_bone=700):
    mesh = o3d.io.read_triangle_mesh("/home/connorscomputer/partial.ply")
    # Load the mesh
    mesh = trimesh.load(ply_path)

    # Calculate bounds for voxel grid
    bounds = mesh.bounds
    dims = np.ceil((bounds[1] - bounds[0]) / voxel_size).astype(int)

    # Create voxel grid
    voxels = mesh.voxelized(voxel_size)
    volume = voxels.matrix.astype(float)

    # Assign HU values (700 for bone, -1000 for air)
    volume = volume * (hu_bone + 1000) - 1000

    volume = volume.astype(np.int16)

    affine = np.eye(4)
    affine[:3, :3] = np.diag([voxel_size] * 3)

    # Create and save NIFTI
    nifti_image = nib.Nifti1Image(volume, affine)
    nib.save(nifti_image, output_path)


if __name__ == "__main__":
    input_path = "/home/connorscomputer/partial.ply"
    output_path = "spine_ct.nii.gz"
    mesh_to_ct(input_path, output_path)
