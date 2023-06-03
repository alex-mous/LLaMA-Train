import os
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.main.llama import Transformer, Tokenizer, ModelArgs
from src.main.util import get_pile_dataloader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_path: str = os.path.dirname(__file__).removesuffix(os.path.normpath("/src/main/train"))


def compute_loss(
        model: nn.Module,
        tokens: torch.Tensor,
        loss_fn: nn.CrossEntropyLoss
) -> torch.Tensor:
    logits = model(tokens[:, :-1])
    flattened_logits = logits.reshape(-1, model.params.vocab_size)
    shift_tokens = tokens[:, 1:].reshape(-1).long()
    loss = loss_fn(flattened_logits, shift_tokens)
    return loss


def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        weight_decay: float,
        clip: float = 1.0
):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader):
            tokens = batch.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, tokens, loss_fn)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()


def main():
    # load dataset.
    batch_size: int = 32
    train_loader, val_loader = get_pile_dataloader(batch_size=batch_size)
    # load model.
    tokenizer = Tokenizer()
    args = ModelArgs()
    args.vocab_size = tokenizer.vocab_size
    args.num_layers = 2
    model = Transformer(args)

    num_params = 0
    for name, param in model.named_parameters():
        print(name, param.numel())
        num_params += param.numel()
    print(f"Number of parameters = {num_params}")

    # train model.
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        lr=8.0e-2,
        weight_decay=0.1
    )


if __name__ == "__main__":
    main()
