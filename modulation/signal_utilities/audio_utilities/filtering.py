#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11-01-2021
           """

import numpy
import torch
from scipy.signal import hilbert, lfilter
from warg import next_pow_2

__all__= ['mel_scale','in_mel_scale','pre_emphasise','pre_emphasise_torch','de_emphasise',"fft_frequencies","hilbert_envelope"]

def mel_scale(x: numpy.ndarray) -> numpy.ndarray:
    """

    :param x:
    :type x:
    :return:
    :rtype:
    """
    return 2595.0 * numpy.log10(1.0 + x / 700.0)


def in_mel_scale(x: numpy.ndarray) -> numpy.ndarray:
    """

    :param x:
    :type x:
    :return:
    :rtype:
    """
    return 700 * (10 ** (x / 2595.0) - 1)


def pre_emphasise(x: numpy.ndarray, coeff: float = 0.97) -> numpy.ndarray:
    """

    :param x:
    :type x:
    :param coeff:
    :type coeff:
    :return:
    :rtype:
    """
    return lfilter([1, -coeff], [1], x)


def de_emphasise(x: numpy.ndarray, coeff: float = 0.97) -> numpy.ndarray:
    """

    :param x:
    :type x:
    :param coeff:
    :type coeff:
    :return:
    :rtype:
    """
    return lfilter([1], [1, -coeff], x)


def pre_emphasise_torch(signal: numpy.ndarray, coeff: float = 0.97) -> numpy.ndarray:
    """
    Pre-emphasis on the input signal
    :param signal: (time,)
    :param coeff:
    :return: (time,)
    """
    return torch.cat((signal[0:1], signal[1:] - coeff * signal[:-1]))


def fft_frequencies(sr: int = 16000, n_fft: int = 512) -> numpy.ndarray:
    """

    :param sr:
    :type sr:
    :param n_fft:
    :type n_fft:
    :return:
    :rtype:
    """
    return numpy.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)


def hilbert_envelope(signal: numpy.ndarray, *, n_fft: int = None) -> numpy.ndarray:
    """Calculates the Hilbert envelope of a signal.

    :param signal: The signal to calculate the Hilbert envelope of.
          Signal on which to calculate the hilbert envelope. The calculation
        is done along the last axis (i.e. ``axis=-1``).
    :param n_fft: The FFT size to use. If not provided, it will be determined
        from the length of the signal.
    :return: The Hilbert envelope of the signal.


    """
    signal_length = signal.shape[-1]
    if not n_fft:
        n_fft = next_pow_2(signal_length)
    return numpy.abs(
        hilbert(signal, n_fft)[..., :signal_length]
    )  # Return signal with same shape as original


if __name__ == "__main__":
    # print(mel_scale(numpy.arange(9)**2))
    # print(len(fft_frequencies())) # oneside
    # print(len(numpy.fft.fftfreq(512)))
    def aijsda():
        """description"""
        a = list(range(1, 9 + 1))
        b = pre_emphasise(a)
        c = de_emphasise(b)
        assert numpy.allclose(a, c), print(a, "\n", c)

        aijsda()
