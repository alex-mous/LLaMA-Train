"""
Load and save checkpoints for model and optimizer
"""

import os

import torch
from torch import nn, optim


def save_checkpoint(optimizer: optim.Optimizer, model: nn.Module, checkpoint_path: str):
    """
    Save checkpoint from optimizer and model into checkpoint_path

    :param optimizer: Optimizer used during training
    :param model: PyTorch model
    :param checkpoint_path: Checkpoint path and name
    """
    if checkpoint_path is not None:
        torch.save({
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
        }, checkpoint_path)


def load_checkpoint(optimizer: optim.Optimizer, model: nn.Module, checkpoint_path: str):
    """
    Load checkpoint into optimizer and model from checkpoint_path

    :param optimizer: Optimizer used during training (optional)
    :param model: PyTorch model
    :param checkpoint_path: Checkpoint path and name
    """
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])


def generate_checkpoint_name(checkpoints_base_path: str, model: nn.Module, epoch: int):
    """
    Generate a checkpoint name for the given model type and epoch

    :param checkpoints_base_path: Path to directory to store checkpoints in
    :param model: PyTorch model
    :param epoch: Epoch number
    :return: Checkpoint path and name
    """
    return os.path.join(checkpoints_base_path, f"chkpt-{model.__class__.__name__}-{epoch}.pt")
