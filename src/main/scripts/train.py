import torch
import gc
import os
from tqdm import tqdm

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


def main():
    # Model params
    dim = 64
    n_layers = 2
    n_heads = 2

    # Data params
    data_path = os.path.join(root_dir, os.path.normpath("data/"))
    max_seq_len = 32
    n_train = 4_000_000
    n_val = 10_000
    batch_size = 128

    # Training params
    output_path = os.path.join(root_dir, os.path.normpath("output/"))
    epochs = 2
    lr = 3e-4
    weight_decay = 0.01

    # Checkpoints
    ichkpt_path = None  # initial checkpoint, if any
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
        new_chkpt_format=True
    )

    try:
        # Train model
        print("Training model...")
        train(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            lr=lr,
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


if __name__ == "__main__":
    main()
