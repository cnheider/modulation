#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 18-11-2020
           """

from draugr.numpy_utilities import SplitEnum

from modulation.data.audio.speech.recognition.speech_command import SpeechCommands


def labels_index_test():
  """description"""
  train_set = SpeechCommands()

  word_start = "yes"
  index = train_set.label_to_index(word_start)
  word_recovered = train_set.index_to_label(index)

  print(word_start, "-->", index, "-->", word_recovered)


def labels_index_same_test():
  """description"""
  train_set = SpeechCommands()
  valid_set = SpeechCommands(split = SplitEnum.validation)
  test_set = SpeechCommands(split = SplitEnum.testing)

  word_start = "yes"
  index_train = train_set.label_to_index(word_start)
  word_recovered_train = train_set.index_to_label(index_train)

  index_valid = valid_set.label_to_index(word_start)
  word_recovered_valid = valid_set.index_to_label(index_valid)

  index_test = test_set.label_to_index(word_start)
  word_recovered_test = test_set.index_to_label(index_test)

  assert index_test == index_train == index_valid
  assert word_recovered_train == word_recovered_valid == word_recovered_test


if __name__ == "__main__":
  labels_index_test()
  labels_index_same_test()
