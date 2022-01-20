#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03-12-2020
           """

import numpy
import numpy
import torch
from matplotlib import pyplot


def one_sample_positive_class_precisions(
    scores: numpy.array, truths: numpy.array
) -> numpy.array:
    """

    Reference implementation of l$\omega$lrap both natively and using sklearn.metrics.

      Calculate precisions for each true class for a single sample.

      Args:
        scores: numpy.array of (num_classes,) giving the individual classifier scores.
        truths: numpy.array of (num_classes,) bools indicating which classes are true.

      Returns:
        pos_class_indices: numpy.array of indices of the true classes for this sample.
        pos_class_precisions: numpy.array of precisions corresponding to each of those
          classes.
    """

    pos_class_indices = numpy.flatnonzero(truth > 0)

    if not len(
        pos_class_indices
    ):  # Only calculate precisions if there are some true classes.
        return pos_class_indices, numpy.zeros(0)

    retrieved_classes = numpy.argsort(scores)[
        ::-1
    ]  # Retrieval list of classes for this sample.

    num_classes = scores.shape[0]
    class_rankings = numpy.zeros(
        num_classes, dtype=numpy.int
    )  # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings[retrieved_classes] = range(num_classes)

    retrieved_class_true = numpy.zeros(
        num_classes, dtype=numpy.bool
    )  # Which of these is a true label?
    retrieved_class_true[class_rankings[pos_class_indices]] = True

    retrieved_cumulative_hits = numpy.cumsum(
        retrieved_class_true
    )  # Num hits for every truncated retrieval list.
    precision_at_hits = retrieved_cumulative_hits[  # Precision of retrieval list truncated at each hit, in order of pos_labels.
        class_rankings[pos_class_indices]
    ] / (
        1 + class_rankings[pos_class_indices].astype(numpy.float)
    )

    return pos_class_indices, precision_at_hits


def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = numpy.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = numpy.flatnonzero(sample_weight > 0)
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0,
        scores[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices],
    )
    return overall_lwlrap


class lwlrap_accumulator(object):
    """Accumulate batches of test samples into per-class and overall lwlrap."""

    def __init__(self):
        self.num_classes = 0
        self.total_num_samples = 0

    def accumulate_samples(self, batch_truth, batch_scores):
        """Cumulate a new batch of samples into the metric.

        Args:
          truth: numpy.array of (num_samples, num_classes) giving boolean
            ground-truth of presence of that class in that sample for this batch.
          scores: numpy.array of (num_samples, num_classes) giving the
            classifier-under-test's real-valued score for each class for each
            sample.
        """
        assert batch_scores.shape == batch_truth.shape
        num_samples, num_classes = batch_truth.shape
        if not self.num_classes:
            self.num_classes = num_classes
            self._per_class_cumulative_precision = numpy.zeros(self.num_classes)
            self._per_class_cumulative_count = numpy.zeros(
                self.num_classes, dtype=numpy.int
            )
        assert num_classes == self.num_classes
        for truth, scores in zip(batch_truth, batch_scores):
            pos_class_indices, precision_at_hits = one_sample_positive_class_precisions(
                scores, truth
            )
            self._per_class_cumulative_precision[pos_class_indices] += precision_at_hits
            self._per_class_cumulative_count[pos_class_indices] += 1
        self.total_num_samples += num_samples

    def per_class_lwlrap(self):
        """Return a vector of the per-class lwlraps for the accumulated samples."""
        return self._per_class_cumulative_precision / numpy.maximum(
            1, self._per_class_cumulative_count
        )

    def per_class_weight(self):
        """Return a normalized weight vector for the contributions of each class."""
        return self._per_class_cumulative_count / float(
            numpy.sum(self._per_class_cumulative_count)
        )

    def overall_lwlrap(self):
        """Return the scalar overall lwlrap for cumulated samples."""
        return numpy.sum(self.per_class_lwlrap() * self.per_class_weight())


def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: numpy.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: numpy.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: numpy.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: numpy.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply numpy.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = numpy.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = one_sample_positive_class_precisions(
            scores[sample_num, :], truth[sample_num, :]
        )
        precisions_for_samples_by_classes[
            sample_num, pos_class_indices
        ] = precision_at_hits
    labels_per_class = numpy.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(numpy.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = numpy.sum(
        precisions_for_samples_by_classes, axis=0
    ) / numpy.maximum(1, labels_per_class)
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = numpy.sum(precisions_for_samples_by_classes) / numpy.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = numpy.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


if __name__ == "__main__":
    # Random test data.
    num_samples = 100
    num_labels = 20

    truth = numpy.random.rand(num_samples, num_labels) > 0.5
    # Ensure at least some samples with no truth labels.
    truth[0:1, :] = False

    scores = numpy.random.rand(num_samples, num_labels)

    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(truth, scores)
    print(
        "lwlrap from per-class values=", numpy.sum(per_class_lwlrap * weight_per_class)
    )
    print(
        "lwlrap from sklearn.metrics =", calculate_overall_lwlrap_sklearn(truth, scores)
    )

    # Test of accumulator version.
    accumulator = lwlrap_accumulator()
    batch_size = 12
    for base_sample in range(0, scores.shape[0], batch_size):
        accumulator.accumulate_samples(
            truth[base_sample : base_sample + batch_size, :],
            scores[base_sample : base_sample + batch_size, :],
        )
    print("cumulative_lwlrap=", accumulator.overall_lwlrap())
    print("total_num_samples=", accumulator.total_num_samples)
