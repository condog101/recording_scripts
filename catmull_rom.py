import numpy as np
from splines import CatmullRom
from geomdl import BSpline, fitting
# from geomdl.visualization import VisMPL


def fit_catmull_rom(points, alpha=0, sample_points=300, bspline=True):

    # this assumes points are passed in correct order, otherwise doesn't work
    # Create spline

    if bspline:
        spline = fit_weighted_bspline(points)
        spline.sample_size = sample_points
        return np.array(spline.evalpts)
    else:

        diffs = np.diff(points, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        t = np.concatenate(([0], np.cumsum(distances)))
        t = t / t[-1]  # normalize to [0,1]
        spline = CatmullRom(points, grid=t, endconditions='natural')

        # Sample 100 points evenly in parameter space
        sample_t = np.linspace(0, spline.grid[-1], sample_points)
        return spline.evaluate(sample_t)


def calculate_point_weights(points, k=6):
    """
    Calculate weights for each point based on isolation.
    Points that are further from their k nearest neighbors get higher weights.

    Args:
        points (np.array): Array of 3D points
        k (int): Number of nearest neighbors to consider

    Returns:
        np.array: Weights for each point
    """
    n_points = len(points)
    weights = np.zeros(n_points)

    for i in range(n_points):
        # Calculate distances to all other points
        distances = np.linalg.norm(points - points[i], axis=1)
        # Sort distances and take k nearest neighbors (excluding self)
        nearest_distances = np.sort(distances)[1:k+1]
        # Weight is average distance to k nearest neighbors
        weights[i] = np.mean(nearest_distances)

    # Normalize weights to [0.1, 1.0] range
    weights = (weights - weights.min()) / \
        (weights.max() - weights.min()) * 0.9 + 0.1
    return weights


def fit_weighted_bspline(points, degree=3):
    """
    Fit a B-spline curve to 3D points with distance-based weights.

    Args:
        points (np.array): Array of 3D points
        degree (int): Degree of the B-spline curve
        num_ctrlpts (int): Number of control points for the curve

    Returns:
        geomdl.BSpline.Curve: Fitted B-spline curve
    """
    # Convert points to numpy array if not already
    points = np.array(points)

    # Calculate weights based on point isolation
    weights = calculate_point_weights(points)

    # Fit the curve using weighted least squares
    curve = fitting.approximate_curve(
        points.tolist(),
        degree,
        # weights=weights.tolist()
    )

    # Set visualization settings

    return curve
