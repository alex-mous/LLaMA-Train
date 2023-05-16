import torch
from torch import nn
from torch.utils.data import DataLoader


def accuracy(model: nn.Module, dataset_loader: DataLoader, device: torch.device) -> float:
    pass
