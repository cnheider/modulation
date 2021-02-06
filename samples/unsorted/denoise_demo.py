#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

            Removes all fourier coefs below threshold in frequency space

           Created on 07/04/2020
           """

import numpy
from matplotlib import pyplot

from neodroidaudition.regression.spectral_denoise import fft_denoise

if __name__ == "__main__":

    def main():
        """"""
        delta = 0.001
        time_ = numpy.arange(0, 1, delta)
        time_steps = len(time_)

        #
        signal = numpy.sin(2 * numpy.pi * 50 * time_) + numpy.sin(
            2 * numpy.pi * 120 * time_
        )
        noisy_signal = signal + 2.5 * numpy.random.randn(time_steps)

        cleaned_signal = fft_denoise(noisy_signal, time_steps)
        #
        fig, axs = pyplot.subplots(4, 1)
        frequencies = (1 / (delta * time_steps)) * numpy.arange(time_steps)
        L = numpy.arange(1, numpy.floor(time_steps / 2), dtype=numpy.int)

        #
        pyplot.sca(axs[0])
        pyplot.plot(time_, noisy_signal, color="c", LineWidth=1.5, label="Noisy")
        pyplot.plot(time_, signal, color="k", LineWidth=2, label="Clean")
        pyplot.xlim(time_[0], time_[-1])
        pyplot.ylabel("Amplitude")
        pyplot.xlabel("Time")
        pyplot.legend()
        #
        """
        psd_cleaned = power_spectral_density * indices
      cleaned_f_coef = noisy_signal_f_coef * indices

    pyplot.sca(axs[1])
    pyplot.plot(frequencies[L], power_spectral_density[L], color='c', LineWidth=2, label='Noisy')
    pyplot.xlim(frequencies[L[0]], frequencies[L[-1]])
    pyplot.ylabel('PSD')
    pyplot.xlabel('Frequency')
    pyplot.legend()
    #
    pyplot.sca(axs[2])
    pyplot.plot(frequencies[L], psd_cleaned[L], color='c', LineWidth=2, label='Cleaned')
    pyplot.xlim(frequencies[L[0]], frequencies[L[-1]])
    pyplot.ylabel('PSD')
    pyplot.xlabel('Frequency')
    pyplot.legend()
      """
        #
        pyplot.sca(axs[3])
        pyplot.plot(time_, noisy_signal, color="g", LineWidth=1, label="Noisy")
        pyplot.plot(time_, cleaned_signal, color="c", LineWidth=3, label="Cleaned")
        pyplot.plot(time_, signal, color="k", LineWidth=2, label="Clean")
        pyplot.xlim(time_[0], time_[-1])
        pyplot.ylabel("Amplitude")
        pyplot.xlabel("Time")
        pyplot.legend()
        #
        pyplot.show()

    main()
