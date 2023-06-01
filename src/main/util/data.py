"""
Data processing and loading
"""

import numpy as np
import os
import time
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
    Loaded from an array of sequences, each of same length (seq_len)
    """
    def __init__(self, seqs: torch.tensor):
        self.seqs = seqs

    def __getitem__(self, idx):
        if idx >= self.__len__():
            return None
        return {
            "input_ids": self.seqs[idx]
        }

    def __len__(self):
        return self.seqs.shape[0]


def process_file(datafile: str, max_seqs: int = 20000, seq_len: int = 2048) -> torch.tensor:
    """
    Process JSONL file into up to max_seqs seqs of tokens of length seq_len
    Returns tensor of sequences, each of seq_len with no padding
    :param datafile:
    :param max_seqs:
    :param seq_len:
    :return: Tensor of dimension (up to max_seqs, seq_len)
    """
    if max_seqs == 0:
        return None

    seqs = torch.zeros((1, seq_len), dtype=torch.int32)

    tokenizer = Tokenizer()
    with open(datafile, "r", encoding="utf-8") as file:
        with tqdm(total=max_seqs) as p_bar:
            curr_tokens = torch.tensor([], dtype=torch.int32)
            for jsonline in file:
                if seqs.shape[0] > max_seqs:
                    break
                line = json.loads(jsonline)
                new_tokens = torch.tensor(tokenizer.encode(line["text"]), dtype=torch.int32)
                curr_tokens = torch.cat((curr_tokens, new_tokens))
                if len(curr_tokens) > seq_len:
                    tokens = torch.reshape(curr_tokens[:-(len(curr_tokens) % seq_len)], (-1, seq_len))
                    curr_tokens = curr_tokens[tokens.shape[0]*seq_len : ]
                    seqs = torch.vstack((seqs, tokens))
                    p_bar.update(tokens.shape[0])
    return seqs[1:]


def load_pile(num_train: int = 20000, num_val: int = 10000, num_test: int = 0, seq_len: int = 2048):
    """
    Load Pile dataset into train, val, and test datasets of tokens
    with numbers of sequences and sequence lengths as specified
    :param num_train:
    :param num_val:
    :param num_test:
    :param seq_len:
    :return: train, test, val PileDatasets
    """
    print("Loading Pile dataset...")
    start_time = time.time()

    train_toks = process_file(os.path.join(data_path, "train.jsonl"), num_train, seq_len)
    val_toks = process_file(os.path.join(data_path, "val.jsonl"), num_val, seq_len)
    test_toks = process_file(os.path.join(data_path, "test.jsonl"), num_test, seq_len)
    train = PileDataset(train_toks)
    val = PileDataset(val_toks)
    test = PileDataset(test_toks)

    print(f"Loaded dataset in {time.time() - start_time:.2f} seconds")
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
