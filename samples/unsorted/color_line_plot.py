#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 8/2/22
           """

__all__ = []

import numpy
from matplotlib import pyplot

from modulation.visualisation.colorise_waveform import color_plot


def main():
  """description"""
  lens = 9999
  data = (numpy.random.random((lens)) - 0.5).cumsum(axis = 0)
  xy = numpy.vstack((numpy.arange(lens), data)).T
  print(xy.shape)

  color_plot(xy)
  pyplot.show()


if __name__ == "__main__":
  main()
