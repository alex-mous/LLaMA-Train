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


def get_number_of_parameters(model: nn.Module):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            total_params += parameter.numel()
    return total_params
