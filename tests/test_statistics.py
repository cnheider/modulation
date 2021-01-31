#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 17-12-2020
           '''

from statistics import mean

from neodroidaudition.audio_utilities.signal_statistics import root_mean_square


def test_root_mean_square_signed():
  s = [i - 3 for i in range(6)]
  a = root_mean_square(s)
  print(a)
  assert a


def test_root_mean_square_unsigned():
  s = [i + 3 for i in range(6)]
  assert root_mean_square(s) == mean(s)
