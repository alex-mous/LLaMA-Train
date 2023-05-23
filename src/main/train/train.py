import argparse
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.main.util import get_data_loader, process_data_to_txt, load_pile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_path: str = os.path.dirname(__file__).removesuffix(os.path.normpath("/src/main/train"))


def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int
) -> List[float]:
    model.to(device)
    train_losses = []
    # TODO: implement train

    return train_losses


def main(*args, **kwargs):
    # TODO: setup data
    train_loader, val_loader, test_loader = get_data_loader()

    # TODO: load model based on checkpoints, params from example code
    # llama_model = load(   # probably shouldn't use inference load for train.
    #
    # )

    print(f"Training model")
    # Train model
    train_losses, train_accuracies = train(
        train_loader,
        val_loader,
        **kwargs
    )


def preprocess_data():
    # Process datasets to text files and train tokenizer
    artifacts_path = os.path.join(base_path, os.path.join("data"))
    train_text_path = os.path.join(artifacts_path, "07_medium.txt")
    if os.path.exists(train_text_path):
        print(f"File \"{train_text_path}\" is already loaded.")
    else:
        train_data, val_data, test_data = load_pile()
        process_data_to_txt(train_data, train_text_path, p=1e-2)


def run_main():
    arg_parser = argparse.ArgumentParser()
    args, _ = arg_parser.parse_known_args()  # Only parse known args
    main(**vars(args))


if __name__ == "__main__":
    preprocess_data()
    # run_main()
