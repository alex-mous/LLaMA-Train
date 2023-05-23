import argparse
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.main.inference import load
from src.main.util import get_data_loader, process_data_to_txt, load_pile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_path: str = os.path.dirname(__file__).rstrip(os.path.normpath("/src/main/train/train.py"))


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
    llama_model = load(

    )

    print(f"Training model")
    # Train model
    train_losses, train_accuracies = train(
        train_loader,
        val_loader,
        **kwargs
    )


def preprocess_data():
    # Process datasets to text files and train tokenizer
    artifacts_path = os.path.join(base_path, os.path.join("artifacts"))
    train, val, test = load_pile()
    process_data_to_txt(train, os.path.join(artifacts_path, "07.txt"))
    process_data_to_txt(val, os.path.join(artifacts_path, "val.txt"))
    process_data_to_txt(test, os.path.join(artifacts_path, "test.txt"))


def run_main():
    arg_parser = argparse.ArgumentParser()

    args, _ = arg_parser.parse_known_args()  # Only parse known args
    main(**vars(args))


if __name__ == "__main__":
    preprocess_data()
    #run_main()
