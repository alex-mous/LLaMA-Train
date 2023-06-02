"""
Data processing and loading
"""

import json
import os
import time

import torch
from torch.utils.data import Dataset, DataLoader

from src.main.llama import Tokenizer


data_path: str = os.path.join(
    os.path.dirname(__file__).removesuffix(os.path.normpath("src/main/util")),
    os.path.normpath("data/")
)


class PileDataset(Dataset):
    """
    Dataset for loading the Pile dataset.
    Loaded from an array of sequences, each of equal length.
    """
    def __init__(self, text: torch.Tensor):
        self.text = text

    def __getitem__(self, idx):
        return {
            "input_ids": self.text[idx]
        }

    def __len__(self):
        return len(self.text)


def process_file(
        file_path: str,
        max_samples: int = 200000
):
    # check if corresponding artifact exists.
    artifact_path = f"{os.path.splitext(file_path)[0]}.pt"
    if os.path.isfile(artifact_path):
        print(f"Artifact found. Loading dataset from {artifact_path}")
        return torch.load(artifact_path)
    # otherwise, parse file.
    print(f"No artifact found. Loading dataset from {file_path}.")
    tokenizer = Tokenizer()
    tokens_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for json_line in file:
            line = json.loads(json_line)
            tokens = torch.tensor(tokenizer.encode(line["text"]), dtype=torch.int32)
            tokens_list.append(tokens)
            if len(tokens_list) >= max_samples:
                break
    # save artifact.
    torch.save(tokens_list, artifact_path)
    # return raw inputs.
    return tokens_list


def convert_file_to_dataset(
        file_path: str,
        num_samples: int = None,
        seq_len: int = 2048,
):
    # load tokens from file path.
    tokens_list = process_file(file_path)
    if num_samples is not None:
        tokens_list = tokens_list[:num_samples]
    # wrap tokens to sequence length chunks.
    tokens_cat = torch.cat(tokens_list)
    tokens_cat = tokens_cat[:-(len(tokens_cat) % seq_len)]
    tokens_cat = tokens_cat.reshape(-1, seq_len)
    return PileDataset(tokens_cat)


def load_pile_dataset(
        num_train: int = 20000,
        num_val: int = 10000,
        seq_len: int = 2048
):
    print(f"Loading Pile dataset...")
    start_time = time.time()

    train_dataset = convert_file_to_dataset(
        file_path=os.path.join(data_path, "train.jsonl"),
        num_samples=num_train,
        seq_len=seq_len
    )
    val_dataset = convert_file_to_dataset(
        file_path=os.path.join(data_path, "val.jsonl"),
        num_samples=num_val,
        seq_len=seq_len
    )

    print(f"Loaded dataset in {time.time() - start_time:.2f} seconds.")
    return train_dataset, val_dataset


def get_pile_dataloader(batch_size: int = 32):
    train_dataset, val_dataset = load_pile_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
