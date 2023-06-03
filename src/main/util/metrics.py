"""
Compute metrics based on a model and dataset loader
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.main.llama import Transformer


def accuracy(model: nn.Module, dataset_loader: DataLoader, device: torch.device) -> float:
    """
    Calculate accuracy of model based on dataset loader
    :param model:
    :param dataset_loader:
    :param device:
    :return:
    """


def compute_loss(model: Transformer, tokens: torch.tensor, loss_fn : nn.CrossEntropyLoss) -> torch.tensor:
    """
    Compute loss on the input batch of tokens
    :param model:
    :param tokens:
    :param loss_fn:
    :return:
    """
    logits = model.forward(tokens[:, :-1], is_causal=True)
    flattened_logits = logits.reshape(-1, model.params.vocab_size)  # flatten logits for input to cross-entropy loss
    shift_tokens = tokens[:, 1:].reshape(-1)  # shift tokens so we only compute loss after first token
    loss = loss_fn(flattened_logits, shift_tokens)  # compute loss between logits and true tokens, ignoring padding
    return loss


def get_number_of_parameters(model: nn.Module):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            total_params += parameter.numel()
    return total_params
