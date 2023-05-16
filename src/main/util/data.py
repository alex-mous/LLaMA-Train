"""
Data processing and loading
"""

import os

data_path: str = os.path.join(
    os.path.dirname(__file__).rstrip(os.path.normpath("/src/main/util/data.py")),
    "data"
)


def load_pile():
    """
    Load Pile dataset into train, val, and test datasets
    :return:
    """


def get_data_loader():
    """
    Get dataloaders for train, val, and test datasets
    :return:
    """
