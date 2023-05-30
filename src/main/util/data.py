"""
Data processing and loading
"""

import numpy as np
import os
from tqdm.auto import tqdm
import json
from torch.utils.data import DataLoader, Dataset
import torch
from src.main.llama import Tokenizer


data_path: str = os.path.join(
    os.path.dirname(__file__).removesuffix(os.path.normpath("src/main/util")),
    os.path.normpath("data/")
)


class PileDataset(Dataset):
    """
    Dataset for Pile
    Loads data from a jsonl file line-by-line
    """
    def __init__(self, datafile: str):
        self.datafile = datafile

    def __iter__(self):
        with open(self.datafile, "r", encoding="utf-8") as file:
            for jsonline in file:
                line = json.loads(jsonline)
                yield line["text"]

    def __getitem__(self, idx):
        # We do not implement getitem for random access
        return None


def process_file(datafile: str, max_seqs: int = 20000, seq_len: int = 2048) -> torch.tensor:
    """
    Process JSONL file into up to max_seqs seqs of tokens of length seq_len
    :param datafile:
    :param max_seqs:
    :param seq_len:
    :return: Tensor of dimension (up to max_seqs, seq_len)
    """
    seqs = torch.tensor([])
    tokenizer = Tokenizer()
    with open(datafile, "r", encoding="utf-8") as file:
        for jsonline in file and seqs.shape[0] < max_seqs:
            line = json.loads(jsonline)
            tokens = torch.tensor(tokenizer.encode(line))
            tokens = torch.reshape(tokens[:-len(tokens) % seq_len], (-1, 2048))
            torch.vstack((seqs, tokens))
    return seqs


def load_pile():
    """
    Load Pile dataset into train, val, and test datasets
    :return: train, test, val PileDatasets
    """
    train = PileDataset(os.path.join(data_path, "07.jsonl"))
    val = PileDataset(os.path.join(data_path, "val.jsonl"))
    test = PileDataset(os.path.join(data_path, "test.jsonl"))
    return train, val, test


def get_data_loader(batch_size: int = 32):
    """
    Get dataloaders for train, val, and test datasets with given batch size
    :param batch_size: Batch size for all dataloaders
    :return:
    """
    train_set, val_set, test_set = load_pile()
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def process_data_to_txt(dataset: PileDataset, output_file: str, p: float = 1e-4):
    """
    Process dataset to text file and save
    :param p:
    :param dataset:
    :param output_file:
    """
    with open(output_file, "w", encoding="utf-8") as file:
        for line in tqdm(dataset):
            if np.random.rand() < p:
                file.write(line)
