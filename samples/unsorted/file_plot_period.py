#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09-07-2021
           """

import time

import numpy
import pyaudio
import pylab
from modulation import PROJECT_APP_PATH

RATE = 44100
CHUNK = int(RATE / 20)  # RATE / number of updates per second

res_folder = PROJECT_APP_PATH.user_log


def soundplot(stream):
  """

  :param stream:
  :type stream:
  """
  t1 = time.time()
  data = numpy.frombuffer(stream.read(CHUNK), dtype = numpy.int16)
  pylab.plot(data)
  pylab.title(i)
  pylab.grid()
  pylab.axis([0, len(data), -(2 ** 16) / 2, 2 ** 16 / 2])
  pylab.show()
  pylab.savefig(str(res_folder / "03.png"), dpi = 50)
  pylab.close("all")
  print(f"took {(time.time() - t1) * 1000:.02f} ms")


if __name__ == "__main__":
  p = pyaudio.PyAudio()
  stream = p.open(
      format = pyaudio.paInt16,
      channels = 1,
      rate = RATE,
      input = True,
      frames_per_buffer = CHUNK,
      )
  for i in range(int(20 * RATE / CHUNK)):  # do this for 10 seconds
    soundplot(stream)
  stream.stop_stream()
  stream.close()
  p.terminate()
