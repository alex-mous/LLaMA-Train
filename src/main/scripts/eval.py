import argparse
import gc
import os

import torch
from torch.nn import CrossEntropyLoss

from src.main.llama import load_llama_and_data
from src.main.util import evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir: str = os.path.dirname(__file__).removesuffix(os.path.normpath("src/main"))


def model_eval(
        chkpt_path: str,
        data_path: str,
        **args
):
    """
    Train a model with the following parameters
    """

    # Checkpoint to evaluate
    assert os.path.isfile(chkpt_path), "Initial checkpoint required to eval"

    # Load model and dataloaders
    model, tokenizer, _, val_dataloader = load_llama_and_data(
        storage_base_path=data_path,
        tokenizer_path="tokenizer.model",
        train_path="train.jsonl",
        val_path="val.jsonl",
        num_train=10,
        initial_chkpt=chkpt_path,
        **args
    )

    # Perform evaluation
    model.to(device)
    model.eval()
    print(f"Evaluating model stored at {chkpt_path}...")
    loss_fn = CrossEntropyLoss(ignore_index=tokenizer.eos_id)
    print(f"Validation loss on model: {evaluate(device, model, val_dataloader, loss_fn)}")
    model.cpu()


def main():
    """
    Parse args and run evaluation
    """
    arg_parser = argparse.ArgumentParser()
    # paths.
    arg_parser.add_argument(
        "-cp",
        "--chkpt-path",
        help="Path to model checkpoint",
        required=True
    )
    arg_parser.add_argument(
        "-cpt",
        "--new-chkpt-format",
        help="New model checkpoint type (usually true)",
        type=bool,
        default=True
    )
    arg_parser.add_argument(
        "-dp",
        "--data-path",
        help="Path to data folder (containing train.jsonl, val.jsonl, and tokenizer.model)",
        required=True
    )
    # eval params
    arg_parser.add_argument(
        "-n",
        "--num-val",
        help="Number of validation sequences to use",
        default=10_000,
        type=int
    )
    arg_parser.add_argument(
        "-b",
        "--batch-size",
        help="Batch size for validation",
        default=128,
        type=int
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
    model_eval(**vars(args))


if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()
        torch.cuda.empty_cache()
