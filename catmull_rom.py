import numpy as np
from splines import CatmullRom


def fit_catmull_rom(points, alpha=0.5, sample_points=200):

    # this assumes points are passed in correct order, otherwise doesn't work
    # Create spline
    spline = CatmullRom(points, alpha=alpha, endconditions='natural')

    # Sample 100 points evenly in parameter space
    t = np.linspace(0, spline.grid[-1], sample_points)
    return spline.evaluate(t)
