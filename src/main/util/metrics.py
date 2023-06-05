"""
Compute metrics based on a model and dataset loader
"""

import gc
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.main.llama import XFormersTransformer


def evaluate(device: torch.device, model: XFormersTransformer, eval_dataloader: DataLoader, loss_fn: nn.CrossEntropyLoss) -> float:
    """
    Compute loss on eval dataloader
    """
    with torch.no_grad():
        model.eval()
        val_loss = 0
        try:
            for batch in eval_dataloader:
                tokens = batch.to(device)
                val_loss += compute_loss(model, tokens, loss_fn).item()
                tokens.cpu()
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    return val_loss / len(eval_dataloader)


def compute_loss(model: XFormersTransformer, tokens: torch.Tensor, loss_fn: nn.CrossEntropyLoss) -> torch.Tensor:
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
    """
    Get number of parameters in network
    """
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            total_params += parameter.numel()
    return total_params
