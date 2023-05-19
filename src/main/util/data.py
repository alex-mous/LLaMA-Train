"""
Data processing and loading
"""

import os
import json
from torch.utils.data import DataLoader, IterableDataset


data_path: str = os.path.join(
    os.path.dirname(__file__).removesuffix(os.path.normpath("src/main/util")),
    os.path.normpath("data/")
)


class PileDataset(IterableDataset):
    """
    Iterable Dataset for Pile
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


def load_pile():
    """
    Load Pile dataset into train, val, and test datasets
    :return: train, test, val PileDatasets
    """
    train = PileDataset(os.path.join(data_path, "00.jsonl"))
    val = PileDataset(os.path.join(data_path, "val.jsonl"))
    test = PileDataset(os.path.join(data_path, "test.jsonl"))
    return train, val, test


def get_data_loader(batch_size: int = 32):
    """
    Get dataloaders for train, val, and test datasets with given batchsize
    :param batch_size: Batch size for all dataloaders
    :return:
    """
    train_set, val_set, test_set = load_pile()
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader
