#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 8/1/22
           """

__all__ = ["runs_of_non_zero"]

import itertools
from typing import Sequence


def non_zero(x) -> bool:
    return x != 0


def runs_of_non_zero(bits: Sequence) -> Sequence:
    """
    Finds runs of non-zero bits in a sequence.

    :param bits:
    :type bits:
    :return:
    :rtype:

    """
    for bit, group in itertools.groupby(bits, non_zero):
        yield bit, group
