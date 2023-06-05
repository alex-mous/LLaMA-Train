import argparse
import gc
import os
import torch
from tqdm import tqdm
from typing import Optional

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.optim import AdamW

from src.main.llama import XFormersTransformer, Tokenizer, load_llama_and_data
from src.main.util import compute_loss, evaluate
from src.main.util import generate_checkpoint_name

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir: str = os.path.dirname(__file__).removesuffix(os.path.normpath("src/main"))


def train(
        model: XFormersTransformer,
        tokenizer: Tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        weight_decay: float,
        grad_clip: float = 1.0,  # clipping for all gradients
        chkpt_dir: str = None,  # checkpoint directory
        batch_save_freq: int = -1,  # save after this many batches. -1 means only saving at the end of an epoch
):
    # Main training loop, including checkpointing
    if not os.path.isdir(chkpt_dir):
        os.makedirs(chkpt_dir)

    model.to(device)
    model.train()
    loss_fn = CrossEntropyLoss(ignore_index=tokenizer.eos_id)  # ignore padding
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        train_loss = 0
        try:
            for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
                # compute loss between predictions (logits) and tokens
                # each prediction corresponds to the next token, so we shift tokens by one
                tokens = batch.to(device)
                optimizer.zero_grad()
                loss = compute_loss(model, tokens, loss_fn)  # compute logits on all using all but last token
                tokens.cpu()  # ensure tokens can be garbage collected
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                train_loss += loss.item()

                if batch_save_freq > 0 and (i + 1) % batch_save_freq == 0:
                    # Save checkpoint
                    chkpt_path = generate_checkpoint_name(chkpt_dir, f"{epoch + 1}-batch-{i}", True)
                    torch.save(model.state_dict(), chkpt_path)

                    val_loss = evaluate(device, model, val_loader, loss_fn)

                    # Print batch summary
                    print(f"Epoch {epoch + 1}. Batch {i}. Train loss: {train_loss / i}. Val loss: {val_loss}")
        finally:
            # Save checkpoint
            chkpt_path = generate_checkpoint_name(chkpt_dir, f"{epoch + 1}-end", True)
            torch.save(model.state_dict(), chkpt_path)
            # garbage collect to process next batch
            gc.collect()
            torch.cuda.empty_cache()

        val_loss = evaluate(device, model, val_loader, loss_fn)

        # Print summary
        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}. Train loss: {train_loss}. Val loss: {val_loss}")


def run_train(
        data_path: str,
        output_path: str,
        dim: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
        n_train: int,
        n_val: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        ichkpt_path: Optional[str],
        ichkpt_new_chkpt_format: bool,
        **args
):
    """
    Run training with given params
    """

    # Checkpoints
    run_name = f"d{dim}-l{n_layers}-h{n_heads}-seq{max_seq_len}"  # Base directory relative to checkpoints for this run
    chkpts_path = os.path.join(output_path, os.path.normpath(f"checkpoints/{run_name}/"))

    # Save frequency for checkpoints
    batch_save_freq = (n_train // batch_size) / 10  # Save 10 times per epoch

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    model, tokenizer, train_dataloader, val_dataloader = load_llama_and_data(
        storage_base_path=data_path,
        tokenizer_path="tokenizer.model",
        train_path=f"data-seq{max_seq_len}.jsonl",
        val_path="val.jsonl",
        num_train=n_train,
        num_val=n_val,
        batch_size=batch_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        initial_chkpt=ichkpt_path,
        new_chkpt_format=ichkpt_new_chkpt_format,
        **args
    )

    try:
        # Train model
        print("Training model...")
        train(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            lr=learning_rate,
            epochs=epochs,
            weight_decay=weight_decay,
            chkpt_dir=chkpts_path,
            batch_save_freq=batch_save_freq
        )
    finally:
        # Ensure model is on CPU so it can be garbage collected
        print("Cleaning up...")
        model.cpu()
        del model, tokenizer, train_dataloader, val_dataloader
        gc.collect()
        torch.cuda.empty_cache()


def main():
    """
    Parse args and run evaluation
    """
    arg_parser = argparse.ArgumentParser()
    # output/checkpoints
    arg_parser.add_argument(
        "-op",
        "--output-path",
        help="Path to store output",
        required=True
    )
    arg_parser.add_argument(
        "-cp",
        "--ichkpt-path",
        help="Path to initial checkpoint",
        default=None,
        type=str
    )
    arg_parser.add_argument(
        "-icpt",
        "--ichkpt-new-chkpt-format",
        help="Initial checkpoint: new model checkpoint type (usually true)",
        type=bool,
        default=True
    )
    # data params
    arg_parser.add_argument(
        "-dp",
        "--data-path",
        help="Path to data folder (containing train.jsonl, val.jsonl, and tokenizer.model)",
        required=True
    )
    arg_parser.add_argument(
        "-nt",
        "--n-train",
        help="Number of train sequences to use",
        default=100_000,
        type=int
    )
    arg_parser.add_argument(
        "-nv",
        "--n-val",
        help="Number of validation sequences to use",
        default=10_000,
        type=int
    )
    arg_parser.add_argument(
        "-b",
        "--batch-size",
        help="Batch size for train/val",
        default=128,
        type=int
    )
    # training args
    arg_parser.add_argument(
        "-lr",
        "--learning-rate",
        help="Learning rate",
        default=3e-4,
        type=float
    )
    arg_parser.add_argument(
        "-wd",
        "--weight-decay",
        help="Weight decay",
        default=0.01,
        type=float
    )
    arg_parser.add_argument(
        "-n",
        "--epochs",
        help="Number of epochs to train for",
        type=int,
        required=True
    )
    # model args
    arg_parser.add_argument(
        "-sl",
        "--max-seq-len",
        help="Maximum sequence length",
        type=int,
        required=True
    )
    arg_parser.add_argument(
        "-d",
        "--dim",
        help="Transformer dimension",
        type=int,
        required=True
    )
    arg_parser.add_argument(
        "-nl",
        "--n-layers",
        help="Number of transformer layers",
        type=int,
        required=True
    )
    arg_parser.add_argument(
        "-nh",
        "--n-heads",
        help="Number of attention heads",
        type=int,
        required=True
    )
    arg_parser.add_argument(
        "-mo",
        "--multiple-of",
        help="SwiGLU hidden layer multiple",
        default=256,
        type=int
    )
    arg_parser.add_argument(
        "-ne",
        "--norm-eps",
        help="Smoothing value for RMSNorm",
        default=1e-5,
        type=float
    )
    args, _ = arg_parser.parse_known_args()
    run_train(**vars(args))


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        torch.cuda.empty_cache()