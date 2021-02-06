#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-12-2020
           """

import os

import numpy
import soundfile


def audio_read(path, norm=True, start=0, stop=None):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError(f"[{path}] does not exist!")
    try:
        x, sr = soundfile.read(path, start=start, stop=stop)
        if len(x.shape) == 1:  # mono
            if norm:
                rms = (x ** 2).mean() ** 0.5
                scalar = 10 ** (-25 / 20) / rms
                x = x * scalar
            return x, sr
        else:  # multi-channel
            x = x.T
            x = x.sum(axis=0) / x.shape[0]
            if norm:
                rms = (x ** 2).mean() ** 0.5
                scalar = 10 ** (-25 / 20) / rms
                x = x * scalar
            return x, sr
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print("WARNING: Audio type not supported")


def audio_write(data, fs, destpath, norm=False):
    if norm:
        rms = (data ** 2).mean() ** 0.5
        scalar = 10 ** (-25 / 10) / (rms + numpy.eps)
        data = data * scalar
        if max(abs(data)) >= 1:
            data = data / max(abs(data), numpy.eps)

    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    soundfile.write(destpath, data, fs)
    return
