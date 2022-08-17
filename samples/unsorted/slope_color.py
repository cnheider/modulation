#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 8/2/22
           """

__all__ = []

import numpy
from matplotlib import pyplot
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap


def main2():
    """
    Color parts of a line based on its properties, e.g., slope.
    """

    x = numpy.linspace(0, 3 * numpy.pi, 500)
    y = numpy.sin(x)
    z = numpy.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

    # Create a colormap for red, green and blue and a norm to color
    # f' < -0.5 red, f' > 0.5 blue, and the rest green
    cmap = ListedColormap(["r", "g", "b"])
    norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(z)
    lc.set_linewidth(3)

    fig1 = pyplot.figure()
    pyplot.gca().add_collection(lc)
    pyplot.xlim(x.min(), x.max())
    pyplot.ylim(-1.1, 1.1)

    # Now do a second plot coloring the curve using a continuous colormap
    t = numpy.linspace(0, 10, 200)
    x = numpy.cos(numpy.pi * t)
    y = numpy.sin(t)
    points = numpy.array([x, y]).T.reshape(-1, 1, 2)
    segments = numpy.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(
        segments, cmap=pyplot.get_cmap("copper"), norm=pyplot.Normalize(0, 10)
    )
    lc.set_array(t)
    lc.set_linewidth(3)

    fig2 = pyplot.figure()
    pyplot.gca().add_collection(lc)
    pyplot.xlim(-1, 1)
    pyplot.ylim(-1, 1)
    pyplot.show()


if __name__ == "__main__":
    main2()
