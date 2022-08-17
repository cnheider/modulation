#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 15-01-2021
           """

from typing import Sequence, Tuple

import numpy

__all__ = ["mask_split_non_zero_concat", "mask_split_non_zero_segments"]

from modulation.signal_utilities.segmentation import runs_of_non_zero


def mask_split_non_zero_concat(
    mask: Sequence, data: Sequence
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    discards rest of the data if mask is shorter than data
    a is where the mask is 1
    b is where the mask is 0

    :param mask:
    :type mask:
    :param data:
    :type data:
    :return:
    :rtype:
    """
    a = []
    b = []
    for n in range(min(len(data), len(mask))):
        if mask[n] == 0:
            b.append(data[n])
        else:
            a.append(data[n])

    return numpy.asarray(a), numpy.asarray(b)


def mask_split_non_zero_segments(
    mask: Sequence, data: Sequence
) -> Tuple[Sequence, Sequence]:
    """
    discards rest of the data if vad is shorter than data

    a is where the mask is 1
    b is where the mask is 0

    :param mask:
    :type mask:
    :param data:
    :type data:
    :return:
    :rtype:
    """
    data_masked = numpy.zeros_like(data)

    for n in range(min(len(data), len(mask))):
        if mask[n] == 0:
            data_masked[n] = 0
        else:
            data_masked[n] = data[n]

    a = []
    b = []

    for bit, r in runs_of_non_zero(data_masked):
        if bit == 1:
            a.append(list(r))
        else:
            b.append(list(r))

    return a, b


if __name__ == "__main__":
    a = numpy.arange(10)
    b = numpy.arange(10)
    print(mask_split_non_zero_concat(a, b))

    c = numpy.zeros(10)
    c[1:5] = 1
    c[8:10] = 1

    s = range(10)
    print(mask_split_non_zero_segments(c, s))
