import gc
import os

import torch
from torch.nn import CrossEntropyLoss

from src.main.llama import load_llama_and_data
from src.main.util import evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir: str = os.path.dirname(__file__).removesuffix(os.path.normpath("src/main"))


def main(*args, **kwargs):
    # Output path that holds checkpoints folder 
    output_path = os.path.join(root_dir, os.path.normpath("output/"))

    # Data path that stores tokenizer.model
    data_path = os.path.join(root_dir, os.path.normpath("data/"))

    # Model params
    dim = 256
    n_layers = 2
    n_heads = 2
    max_seq_len = 32

    # Eval params
    n_val = 10_000
    batch_size = 128

    # Checkpoint to evaluate
    chkpt_name = "chkpt-1-batch-249-light.pt"
    chkpt_run = f"d{dim}-l{n_layers}-h{n_heads}-seq{max_seq_len}"
    is_new_chkpt = True
    chkpt_name_path = os.path.join(
        output_path,
        os.path.normpath(f"checkpoints/{chkpt_run}/{chkpt_name}")
    )
    assert os.path.isfile(chkpt_name_path), "Initial checkpoint required to eval"

    # Load model and dataloaders
    model, tokenizer, _, val_dataloader = load_llama_and_data(
        storage_base_path=data_path,
        tokenizer_path="tokenizer.model",
        train_path="train.jsonl",
        val_path="val.jsonl",
        num_train=10,
        num_val=n_val,
        batch_size=batch_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        initial_chkpt=chkpt_name_path,
        new_chkpt_format=is_new_chkpt
    )

    # Perform evaluation
    model.to(device)
    model.eval()
    print(f"Evaluadting model stored at {chkpt_name_path}...")
    loss_fn = CrossEntropyLoss(ignore_index=tokenizer.eos_id)
    print(f"Validation loss on model: {evaluate(device, model, val_dataloader, loss_fn)}")
    model.cpu()


if __name__ == "__main__":
    try:
        main()
    finally:
        print("Cleaning up memory...")
        gc.collect()
        torch.cuda.empty_cache()