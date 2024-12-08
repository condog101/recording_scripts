import open3d as o3d
from argparse import ArgumentParser
import numpy as np


def get_axis(points):
    center = points.mean(axis=0)
    centered_points = points - center
    covariance_matrix = np.cov(centered_points.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    x_axis = eigenvectors[0]
    z_unit = np.array([0., 0., -1.])
    min_cand = 1000
    cand = 1
    for ind, vec in enumerate(eigenvectors[1:]):
        cosine = np.dot(vec, z_unit)/(np.linalg.norm(vec)
                                      * np.linalg.norm(z_unit))
        if 1 - cosine < min_cand:
            cand = ind + 1

    z_axis = eigenvectors[cand]
    y_axis = eigenvectors[1 if cand != 1 else 2]
    return x_axis, y_axis, z_axis, center


def flip_normals(normals):
    z_vector = np.array([0, 0, -1])
    flipped_normals = normals * -1.0
    vector_norms = np.linalg.norm(normals, axis=1)
    flipped_vector_norms = np.linalg.norm(flipped_normals, axis=1)
    dot_products = np.dot(normals, z_vector)
    flipped_dot_products = np.dot(flipped_normals, z_vector)
    similarities = dot_products / vector_norms
    flipped_similarities = flipped_dot_products / flipped_vector_norms
    ones = np.ones(shape=similarities.shape)
    similarity_ones = ones - similarities
    flipped_similarity_ones = ones - flipped_similarities
    combined = np.array([normals, flipped_normals])
    selector = np.where(similarity_ones <= flipped_similarity_ones, 0, 1)
    return combined[selector, np.arange(len(selector))]


def main(path):
    pcd = o3d.io.read_point_cloud(path)
    # compute principal components, length wise will be x axis, most similar to z unit vector will be z
    points = np.asarray(pcd.points)
    x_axis, y_axis, z_axis, center = get_axis(points)

    print('normals')
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                             std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(25)
    normals = np.asarray(pcd.normals)
    flipped_normals = flip_normals(normals)
    pcd.normals = o3d.utility.Vector3dVector(flipped_normals)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    o3d.io.write_triangle_mesh(
        "/home/connorscomputer/Documents/pcd_mesh.ply", mesh, write_ascii=False, compressed=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="surgery simulation")
    parser.add_argument(
        "FILE", type=str, help="Path to pcd file")
    args = parser.parse_args()
    filename: str = args.FILE
    main(filename)
