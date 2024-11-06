import numpy as np
from splines import CatmullRom


def fit_catmull_rom(points, alpha=0, sample_points=200):

    # this assumes points are passed in correct order, otherwise doesn't work
    # Create spline

    diffs = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    t = np.concatenate(([0], np.cumsum(distances)))
    t = t / t[-1]  # normalize to [0,1]
    spline = CatmullRom(points, grid=t, endconditions='natural')

    # Sample 100 points evenly in parameter space
    sample_t = np.linspace(0, spline.grid[-1], sample_points)
    return spline.evaluate(sample_t)
