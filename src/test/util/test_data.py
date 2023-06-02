"""
Test data processing and loading
"""

import unittest
from src.main.util.data import *


data_path: str = os.path.join(
    os.path.dirname(__file__).removesuffix(os.path.normpath("src/test/util")),
    os.path.normpath("data/")
)


class TestDataLoading(unittest.TestCase):
    """
    Test data loading from util/data
    """
    def test_process_file(self):
        file_path = f"{data_path}/val.jsonl"
        tokens_list = process_file(file_path=file_path)
        self.assertEqual(200000, len(tokens_list))

    def test_load_pile_dataset(self):
        train, val = load_pile_dataset(num_train=10, num_val=10, seq_len=16)
        self.assertEqual(16, len(train[0]))
        self.assertEqual(16, len(val[0]))

if __name__ == '__main__':
    unittest.main()
