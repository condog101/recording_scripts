import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d


class PCDSurface:

    def __init__(self, points):
        self.tree = cKDTree(points)
        self.points = points

    def fit_local_plane(self, query_point, k=10):
        """Fit a local plane around a query point using k nearest neighbors."""
        # Find k nearest neighbors

        distances, indices = self.tree.query(query_point, k=k)
        neighbors = self.points[indices]

        # Calculate centroid
        centroid = np.mean(neighbors, axis=0)

        # Create the covariance matrix
        cov = np.cov(neighbors.T)

        # Get eigenvectors - normal vector is the one with smallest eigenvalue
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # Corresponds to smallest eigenvalue

        # Ensure normal points "up" (assuming z is up)
        if normal[2] < 0:
            normal = -normal

        return normal, centroid

    def check_point_position(self, query_points, k=10):
        """
        Determine if query points are above or below the approximated surface.
        Returns: array of -1 (below), 0 (on), 1 (above)
        """

        results = np.zeros(len(query_points))

        for i, point in enumerate(query_points):
            # Find nearest surface point
            _, idx = self.tree.query(point, k=1)
            nearest = self.points[idx]

            # Get local plane at nearest point
            normal, centroid = self.fit_local_plane(nearest, k)

            # Vector from centroid to query point
            vec = point - centroid

            # Project vector onto normal to determine position
            dist = np.dot(vec, normal)
            results[i] = np.sign(dist)

        return results


class o3DSurface:
    def __init__(self, pcd):

        self.normals = np.asarray(pcd.normals)
        self.points = np.asarray(pcd.points)
        self.tree = cKDTree(self.points)

    def check_point_position(self, query_points):
        results = np.zeros(len(query_points))
        for i, point in enumerate(query_points):
            _, idx = self.tree.query(point, k=1)
            # go from surface to qury point
            vec = point - self.points[idx]
            dist = np.dot(
                vec, self.normals[idx])
            results[i] = np.sign(dist)
        return results
