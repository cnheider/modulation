#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 24/01/2022
           """

import numpy
import pyaudio  # sudo apt-get install python3-pyaudio
from draugr.drawers import discrete_scroll_plot
from matplotlib import pyplot

RATE = 44100
CHUNK = int(RATE / 20)  # RATE / number of updates per second

if __name__ == "__main__":
  p = pyaudio.PyAudio()
  stream = p.open(
      format = pyaudio.paInt16,
      channels = 1,
      rate = RATE,
      input = True,
      frames_per_buffer = CHUNK,
      )
  window_length = int(RATE / CHUNK)

  anim = discrete_scroll_plot(
      (
          numpy.frombuffer(stream.read(CHUNK), dtype = numpy.int16)
          for s in range(int(20 * window_length))
          ),
      window_length = window_length,
      )

  try:
    pyplot.show()
  except:
    print("Plot Closed")
    stream.stop_stream()
    stream.close()
    p.terminate()
