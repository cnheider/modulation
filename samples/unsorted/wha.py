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


class SWHear(object):
  """
  The SWHear class is made to provide access to continuously recorded
  (and mathematically processed) microphone data.
  """

  def __init__(self, device = None, start_streaming = True):
    """fire up the SWHear class."""
    print(" -- initializing SWHear")

    self.chunk = 4096  # number of data points to read at a time
    self.rate = 44100  # time resolution of the recording device (Hz)

    # for tape recording (continuous "tape" of recent audio)
    self.tapeLength = 2  # seconds
    self.tape = numpy.empty(self.rate * self.tapeLength) * numpy.nan

    self.p = pyaudio.PyAudio()  # start the PyAudio class
    if start_streaming:
      self.stream_start()

  ### LOWEST LEVEL AUDIO ACCESS
  # pure access to microphone and stream operations
  # keep math, plotting, FFT, etc out of here.

  def stream_read(self):
    """return values for a single chunk"""
    data = numpy.fromstring(self.stream.read(self.chunk), dtype = numpy.int16)
    # print(data)
    return data

  def stream_start(self):
    """connect to the audio device and start a stream"""
    print(" -- stream started")
    self.stream = self.p.open(
        format = pyaudio.paInt16,
        channels = 1,
        rate = self.rate,
        input = True,
        frames_per_buffer = self.chunk,
        )

  def stream_stop(self):
    """close the stream but keep the PyAudio instance alive."""
    if "stream" in locals():
      self.stream.stop_stream()
      self.stream.close()
    print(" -- stream CLOSED")

  def close(self):
    """gently detach from things."""
    self.stream_stop()
    self.p.terminate()

  ### TAPE METHODS
  # tape is like a circular magnetic ribbon of tape that's continously
  # recorded and recorded over in a loop. self.tape contains this data.
  # the newest data is always at the end. Don't modify data on the type,
  # but rather do math on it (like FFT) as you read from it.

  def tape_add(self):
    """add a single chunk to the tape."""
    self.tape[: -self.chunk] = self.tape[self.chunk:]
    self.tape[-self.chunk:] = self.stream_read()

  def tape_flush(self):
    """completely fill tape with new data."""
    readsInTape = int(self.rate * self.tapeLength / self.chunk)
    print(
        " -- flushing %d s tape with %dx%.2f ms reads"
        % (self.tapeLength, readsInTape, self.chunk / self.rate)
        )
    for i in range(readsInTape):
      self.tape_add()

  def tape_forever(self, plot_sec = 0.25):
    """

    :param plot_sec:
    :type plot_sec:
    :return:
    :rtype:
    """
    t1 = 0
    try:
      while True:
        self.tape_add()
        if (time.time() - t1) > plot_sec:
          t1 = time.time()
          self.tape_plot()
    except:
      print(" ~~ exception (keyboard?)")
      return

  def tape_plot(self, saveAs = "03.png"):
    """plot what's in the tape."""
    pylab.plot(numpy.arange(len(self.tape)) / self.rate, self.tape)
    pylab.axis([0, self.tapeLength, -(2 ** 16) / 2, 2 ** 16 / 2])
    if saveAs:
      t1 = time.time()
      pylab.savefig(saveAs, dpi = 50)
      print(f"plotting saving took {(time.time() - t1) * 1000:.02f} ms")
    else:
      pylab.show()
      print()  # good for IPython
    pylab.close("all")


if __name__ == "__main__":
  ear = SWHear()
  ear.tape_forever()
  ear.close()
  print("DONE")
