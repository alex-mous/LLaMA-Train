"""
Data processing and loading
"""

from typing import Tuple, Optional
import json
import os
import time
from tqdm import tqdm
import math

import torch
from torch.utils.data import Dataset, DataLoader

from src.main.llama import Tokenizer


default_data_path: str = os.path.join(
    os.path.dirname(__file__).removesuffix(os.path.normpath("src/main/util")),
    os.path.normpath("data/")
)

artifacts_path: str = os.path.join(
    os.path.dirname(__file__).removesuffix(os.path.normpath("src/main/util")),
    os.path.normpath("artifacts/")
)


class PileDataset(Dataset):
    """
    Dataset for loading the Pile dataset.
    Loaded from an array of sequences, each of equal length.
    """
    def __init__(self, seqs: torch.Tensor):
        self.seqs = seqs

    def __getitem__(self, idx):
        if idx >= self.__len__():
            return None
        return self.seqs[idx]

    def __len__(self):
        return self.seqs.shape[0]


def _tokenize_line(line: str, tokenizer: Tokenizer, max_seq_len: int, pad_id: int):
    # Tokenize a string line into at least one sequence of max_seq_len and return tensor of sequences
    line_tokens = torch.tensor(tokenizer.encode(line, bos=True, eos=False)).long()
    tokens = torch.full((math.ceil(len(line_tokens)/max_seq_len)*max_seq_len, ), pad_id, dtype=torch.long)
    tokens[:len(line_tokens)] = line_tokens
    tokens = tokens.view(max_seq_len, -1).t()
    return tokens


def process_file(
        tokenizer: Tokenizer,
        data_file: str,
        max_seqs: int = 20000,
        max_seq_len: int = 2048
) -> torch.Tensor:
    """
    Process JSONL file into up to max_seqs seqs of tokens of length seq_len
    Returns tensor of sequences, each of seq_len with padding of tokenizer eos id
    :param tokenizer:
    :param data_file:
    :param max_seqs:
    :param max_seq_len:
    :return: Tensor of dimension (up to max_seqs, seq_len)
    """

    # check if corresponding artifact exists.
    artifact_path = os.path.join(os.path.normpath(artifacts_path), f"{os.path.splitext(os.path.basename(data_file))[0]}.pt")
    if os.path.isfile(artifact_path):
        print(f"Artifact tokens found. Loading tokenized dataset from {artifact_path}")
        return torch.load(artifact_path)[:max_seqs]
    # otherwise, parse file.
    print(f"No artifact found at {artifact_path}. Loading and tokenizing dataset from {data_file}.")

    # create artifacts dir if they don't exist
    if not os.path.isdir(artifacts_path):
        os.makedirs(artifacts_path)

    pad_id = tokenizer.eos_id  # padding id
    seqs = torch.empty((max_seqs, max_seq_len), dtype=torch.long)  # sequences to parse

    # process data file into tokenized sequences padded to exactly max_seq_len
    curr = 0  # current sequence
    with open(data_file, "r", encoding="utf-8") as file:
        with tqdm(total=max_seqs, desc="Dataset loading: ") as p_bar:
            for jsonline in file:
                if curr >= max_seqs:
                    break
                raw = json.loads(jsonline)
                tokens = _tokenize_line(raw["text"], tokenizer, max_seq_len, pad_id)
                num_toks = min(tokens.shape[0], max_seqs-curr)
                seqs[curr:curr+num_toks, :] = tokens[:num_toks, :]
                curr += num_toks
                p_bar.update(num_toks)

    # save artifact and return
    torch.save(seqs, artifact_path)
    return seqs


def load_pile_dataset(
        tokenizer: Tokenizer,
        train_file : str,
        val_file : str,
        test_file : str = "",
        num_train: int = 20000,
        num_val: int = 10000,
        num_test: int = 0,
        max_seq_len: int = 2048,
        data_path: str = default_data_path
) -> Tuple[PileDataset, PileDataset, Optional[PileDataset]]:
    """
    Load Pile dataset into train, val, and test datasets of tokens
    with numbers of sequences and sequence lengths as specified
    """
    print("Loading Pile dataset...")
    start_time = time.time()

    train_toks = process_file(tokenizer, os.path.join(data_path, train_file), num_train, max_seq_len)
    val_toks = process_file(tokenizer, os.path.join(data_path, val_file), num_val, max_seq_len)
    test = None
    if num_test > 0:
        test_toks = process_file(tokenizer, os.path.join(data_path, test_file), num_test, max_seq_len)
        test = PileDataset(test_toks)
    train = PileDataset(train_toks)
    val = PileDataset(val_toks)

    print(f"Loaded dataset in {time.time() - start_time:.2f} seconds")
    return train, val, test


def get_pile_dataloaders(train_set: PileDataset, val_set: PileDataset, test_set: PileDataset = None, batch_size: int = 32):
    """
    Get dataloaders for train, val, and test datasets with given batch size
    :param train_set:
    :param val_set:
    :param test_set:
    :param batch_size:
    :return:
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
