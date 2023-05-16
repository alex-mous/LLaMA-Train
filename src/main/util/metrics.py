"""
Compute metrics based on a model and dataset loader
"""

import torch
from torch import nn
from torch.utils.data import DataLoader


def accuracy(model: nn.Module, dataset_loader: DataLoader, device: torch.device) -> float:
    """
    Calculate accuracy of model based on dataset loader

    :param model:
    :param dataset_loader:
    :param device:
    :return:
    """
