#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 02-12-2020
           """

import torch
from draugr.numpy_utilities import SplitEnum
from draugr.torch_utilities import (
    TorchEvalSession,
    TorchTrainSession,
    global_torch_device,
)
from draugr.visualisation import progress_bar
from draugr.writers import Writer
from torch.nn import Module
from torch.nn.functional import nll_loss
from torch.utils.data import DataLoader

__all__ = ["single_epoch_fitting", "single_epoch_evaluation"]


def single_epoch_fitting(
    model: torch.nn.Module,
    optimiser,
    train_loader_,
    *,
    epoch: int = None,
    writer: Writer = None,
    device_: torch.device = global_torch_device(),
) -> None:
    """

    :param model:
    :type model:
    :param optimiser:
    :type optimiser:
    :param train_loader_:
    :type train_loader_:
    :param epoch:
    :type epoch:
    :param writer:
    :type writer:
    :param device_:
    :type device_:
    """
    accum_loss = 0
    num_batches = len(train_loader_)

    with TorchTrainSession(model):
        for batch_idx, (data, target) in progress_bar(
            enumerate(train_loader_), description="train batch #", total=num_batches
        ):
            loss = nll_loss(
                model(data.to(device_)).squeeze(), target.to(device_)
            )  # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            accum_loss += loss.item()

    if writer:
        writer.scalar("loss", accum_loss / num_batches, epoch)


def single_epoch_evaluation(
    model: Module,
    evaluation_loader: DataLoader,
    subset: SplitEnum,
    *,
    epoch: int = None,
    writer: Writer = None,
    device: torch.device = global_torch_device(),
) -> float:
    """

    :param model:
    :type model:
    :param evaluation_loader:
    :type evaluation_loader:
    :param subset:
    :type subset:
    :param epoch:
    :type epoch:
    :param writer:
    :type writer:
    :param device:
    :type device:
    :return:
    :rtype:
    """
    correct = 0
    num_batches = len(evaluation_loader)
    with TorchEvalSession(model):
        for data, target in progress_bar(
            evaluation_loader, description=f"{subset} batch #", total=num_batches
        ):
            correct += (
                model(data.to(device))
                .argmax(dim=-1)
                .squeeze()
                .eq(target.to(device))
                .sum()
                .item()
            )

    acc = correct / len(evaluation_loader.dataset)
    if writer:
        writer.scalar(f"{subset}_accuracy", acc, epoch)
    return acc
