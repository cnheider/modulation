#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 18-12-2020
           """

import numpy
from matplotlib import pyplot

t = numpy.linspace(1, 100, 1000)
x_volts = 10 * numpy.sin(t / (2 * numpy.pi))
x_watts = x_volts**2
x_db = 10 * numpy.log10(x_watts)


def a():
    """description"""
    # Signal Generation
    # matplotlib inline

    pyplot.subplot(3, 1, 1)
    pyplot.plot(t, x_volts)
    pyplot.title("Signal")
    pyplot.ylabel("Voltage (V)")
    pyplot.xlabel("Time (s)")
    pyplot.show()

    pyplot.subplot(3, 1, 2)
    pyplot.plot(t, x_watts)
    pyplot.title("Signal Power")
    pyplot.ylabel("Power (W)")
    pyplot.xlabel("Time (s)")
    pyplot.show()

    pyplot.subplot(3, 1, 3)
    pyplot.plot(t, x_db)
    pyplot.title("Signal Power in dB")
    pyplot.ylabel("Power (dB)")
    pyplot.xlabel("Time (s)")
    pyplot.show()


def noise():
    """description"""
    # Adding noise using target SNR

    target_snr_db = 20  # Set a target SNR

    sig_avg_db = 10 * numpy.log10(
        numpy.mean(x_watts)
    )  # Calculate signal power and convert to dB

    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)

    mean_noise = 0  # Generate an sample of white noise
    noise_volts = numpy.random.normal(
        mean_noise, numpy.sqrt(noise_avg_watts), len(x_watts)
    )

    y_volts = x_volts + noise_volts  # Noise up the original signal

    # Plot signal with noise
    pyplot.subplot(2, 1, 1)
    pyplot.plot(t, y_volts)
    pyplot.title("Signal with noise")
    pyplot.ylabel("Voltage (V)")
    pyplot.xlabel("Time (s)")
    pyplot.show()
    # Plot in dB
    y_watts = y_volts**2
    y_db = 10 * numpy.log10(y_watts)
    pyplot.subplot(2, 1, 2)
    pyplot.plot(t, 10 * numpy.log10(y_volts**2))
    pyplot.title("Signal with noise (dB)")
    pyplot.ylabel("Power (dB)")
    pyplot.xlabel("Time (s)")
    pyplot.show()


def spectrogram():
    """description"""
    from scipy import signal
    from matplotlib import pyplot
    import numpy

    # Generate a test signal, a 2 Vrms sine wave whose frequency is slowly modulated around 3kHz, corrupted by white noise of exponentially decreasing magnitude sampled at 10 kHz.

    fs = 10e3
    N = 1e5
    amp = 2 * numpy.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = numpy.arange(N) / float(fs)
    mod = 500 * numpy.cos(2 * numpy.pi * 0.25 * time)
    carrier = amp * numpy.sin(2 * numpy.pi * 3e3 * time + mod)
    noise = numpy.random.normal(scale=numpy.sqrt(noise_power), size=time.shape)
    noise *= numpy.exp(-time / 5)
    x = carrier + noise

    # Compute and plot the STFT’s magnitude.

    f, t, Zxx = signal.stft(x, fs, nperseg=1000)
    pyplot.pcolormesh(t, f, numpy.abs(Zxx), vmin=0, vmax=amp, shading="gouraud")
    pyplot.title("STFT Magnitude")
    pyplot.ylabel("Frequency [Hz]")
    pyplot.xlabel("Time [sec]")
    pyplot.show()

    librosa.display.specshow(
        librosa.amplitude_to_db(Y, ref=numpy.max),
        y_axis="linear",
        x_axis="time",
        sr=Fs,
        hop_length=H,
        cmap="gray_r",
    )


spectrogram()


def known_noise():
    """description"""
    # Adding noise using a target noise power

    target_noise_db = 10  # Set a target channel noise power to something very noisy

    target_noise_watts = 10 ** (target_noise_db / 10)  # Convert to linear Watt units

    mean_noise = 0
    noise_volts = numpy.random.normal(
        mean_noise, numpy.sqrt(target_noise_watts), len(x_watts)
    )  # Generate noise samples

    y_volts = x_volts + noise_volts  # Noise up the original signal (again) and plot

    # Plot signal with noise
    pyplot.subplot(2, 1, 1)
    pyplot.plot(t, y_volts)
    pyplot.title("Signal with noise")
    pyplot.ylabel("Voltage (V)")
    pyplot.xlabel("Time (s)")
    pyplot.show()
    # Plot in dB
    y_watts = y_volts**2
    y_db = 10 * numpy.log10(y_watts)
    pyplot.subplot(2, 1, 2)
    pyplot.plot(t, 10 * numpy.log10(y_volts**2))
    pyplot.title("Signal with noise")
    pyplot.ylabel("Power (dB)")
    pyplot.xlabel("Time (s)")
    pyplot.show()
