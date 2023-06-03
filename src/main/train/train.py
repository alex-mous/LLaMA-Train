from typing import Tuple, Optional
import torch
import time
import gc
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.main.llama import Transformer, Tokenizer, load_llama
from src.main.util import get_pile_dataloaders, load_pile_dataset, compute_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(
        model: Transformer,
        tokenizer: Tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        weight_decay: float,
        grad_clip=1.0
):
    model.to(device)
    model.train()
    loss_fn = CrossEntropyLoss(ignore_index=tokenizer.eos_id)  # ignore padding
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=epochs*len(train_loader))

    for epoch in range(epochs):
        train_loss = 0
        try:
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                # compute loss between predictions (logits) and tokens
                # each prediction corresponds to the next token, so we shift tokens by one
                tokens = batch.to(device)
                optimizer.zero_grad()
                loss = compute_loss(model, tokens, loss_fn)  # compute logits on all using all but last token
                # tokens.cpu()  # ensure tokens can be garbage collected
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                lr_scheduler.step()  # update lr
                train_loss += loss.item()
        finally:
            # garbage collect to process next batch
            gc.collect()
            torch.cuda.empty_cache()

        with torch.no_grad():
            model.eval()
            val_loss = 0
            try:
                for batch in val_loader:
                    tokens = batch.to(device)
                    val_loss += compute_loss(model, tokens, loss_fn).item()
                    tokens.cpu()
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}. Train loss: {train_loss}. Val loss: {val_loss}")

        # TODO: checkpointing


def main():
    # Model, data, and tokenizer arguments
    tokenizer_path = "tokenizer.model"
    train_path = "tiny_train.jsonl"
    val_path = "tiny_val.jsonl"
    assert os.path.isfile(tokenizer_path), "LLaMa tokenizer pretrained model file required"
    assert os.path.isfile(train_path), "Train data subset in JSONL format required"
    assert os.path.isfile(val_path), "Validation data subset in JSONL format required"
    epochs = 20
    batch_size = 16
    lr = 8.0e-2
    weight_decay = 0.1
    max_seq_len = 512
    dim = 512
    n_layers = 4
    n_heads = 4

    torch.cuda.empty_cache()
    model, tokenizer = load_llama(
        tokenizer_path,
        None,
        None,
        max_seq_len=max_seq_len,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads
    )

    train_set, val_set, _ = load_pile_dataset(tokenizer, train_path, val_path, max_seq_len=max_seq_len)
    train_dataloader, val_dataloader, _ = get_pile_dataloaders(train_set, val_set, batch_size=batch_size)

    try:
        train(
            model,
            tokenizer,
            train_dataloader,
            val_dataloader,
            lr=lr,
            epochs=epochs,
            weight_decay=weight_decay
        )
    finally:
        # Ensure model is on CPU so it can be garbage collected
        model.cpu()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Cleanup before exiting
        gc.collect()
        torch.cuda.empty_cache()