#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian"
__doc__ = r"""

           Created on 29/03/2020
           """

import collections
import logging
import os
import os.path
import queue
import wave
from datetime import datetime
from pathlib import Path

import deepspeech
import numpy
import pyaudio
import webrtcvad
from halo import Halo
from scipy import signal

"""



deepspeech~=0.8.0
pyaudio~=0.2.11
webrtcvad~=2.0.10
halo~=0.0.18
numpy>=1.15.1
scipy>=1.1.0
pyautogui~=0.9.52

"""

logging.basicConfig(level=20)


class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    RATE_PROCESS = 16000  # Network/VAD rate-space
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            """

            :param in_data:
            :type in_data:
            :param frame_count:
            :type frame_count:
            :param time_info:
            :type time_info:
            :param status:
            :type status:
            :return:
            :rtype:
            """
            # pylint: disable=unused-argument
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return None, pyaudio.paContinue

        if callback is None:
            callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            "format": self.FORMAT,
            "channels": self.CHANNELS,
            "rate": self.input_rate,
            "input": True,
            "frames_per_buffer": self.block_size_input,
            "stream_callback": proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs["input_device_index"] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, "rb")

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = numpy.fromstring(string=data, dtype=numpy.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = numpy.array(resample, dtype=numpy.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(), input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        """description"""
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(
        lambda self: 1000 * self.block_size // self.sample_rate
    )

    def write_wav(self, filename, data):
        """

        :param filename:
        :type filename:
        :param data:
        :type data:
        """
        logging.info("write wav %s", filename)
        wf = wave.open(filename, "wb")
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
        Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
        Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                  |---utterence---|        |---utterence---|
        """
        if frames is None:
            frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()


def main(ARGS):
    """

    :param ARGS:
    :type ARGS:
    """
    if os.path.isdir(ARGS.model):  # Load DeepSpeech model
        model_dir = ARGS.model
        ARGS.model = os.path.join(model_dir, "output_graph.pb")
        ARGS.scorer = os.path.join(model_dir, ARGS.scorer)

    print("Initializing model...")
    logging.info("ARGS.model: %s", ARGS.model)
    model = deepspeech.Model(ARGS.model)
    if ARGS.scorer:
        logging.info("ARGS.scorer: %s", ARGS.scorer)
        model.enableExternalScorer(ARGS.scorer)

    # Start audio with VAD
    vad_audio = VADAudio(
        aggressiveness=ARGS.vad_aggressiveness,
        device=ARGS.device,
        input_rate=ARGS.rate,
        file=ARGS.file,
    )
    print("Listening (ctrl-C to exit)...")
    frames = vad_audio.vad_collector()

    # Stream from microphone to DeepSpeech using VAD
    spinner = None
    if not ARGS.nospinner:
        spinner = Halo(spinner="line")
    stream_context = model.createStream()
    wav_data = bytearray()
    for frame in frames:
        if frame is not None:
            if spinner:
                spinner.start()
            logging.debug("streaming frame")
            stream_context.feedAudioContent(numpy.frombuffer(frame, numpy.int16))
            if ARGS.savewav:
                wav_data.extend(frame)
        else:
            if spinner:
                spinner.stop()
            logging.debug("end utterence")
            if ARGS.savewav:
                vad_audio.write_wav(
                    os.path.join(
                        ARGS.savewav,
                        datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav"),
                    ),
                    wav_data,
                )
                wav_data = bytearray()
            text = stream_context.finishStream()
            print(f"Recognized: {text}")
            if ARGS.keyboard:
                from pyautogui import typewrite

                typewrite(text)
            stream_context = model.createStream()


if __name__ == "__main__":
    DEFAULT_SAMPLE_RATE = 16000

    import argparse

    parser = argparse.ArgumentParser(
        description="Stream from microphone to DeepSpeech using VAD"
    )

    parser.add_argument(
        "-v",
        "--vad_aggressiveness",
        type=int,
        default=3,
        help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3",
    )
    parser.add_argument("--nospinner", action="store_true", help="Disable spinner")
    parser.add_argument(
        "-w", "--savewav", help="Save .wav files of utterences to given directory"
    )
    parser.add_argument(
        "-f",
        "--file",
        # default=Path.home()/'Downloads'/'audio'/'2830-3980-0043.wav',
        help="Read from .wav file instead of microphone",
    )

    parser.add_argument(
        "-m",
        "--model",
        # required=True,
        default=str(Path.home() / "Downloads" / "deepspeech-0.9.1-models.pbmm"),
        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)",
    )
    parser.add_argument(
        "-s",
        "--scorer",
        default=str(Path.home() / "Downloads" / "deepspeech-0.9.1-models.scorer"),
        help="Path to the external scorer file.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=None,
        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.",
    )
    parser.add_argument(
        "-k",
        "--keyboard",
        action="store_true",
        help="Type output through system keyboard",
    )
    ARGS = parser.parse_args()
    if ARGS.savewav:
        os.makedirs(ARGS.savewav, exist_ok=True)
    main(ARGS)
