"""
Test data processing and loading
"""

import unittest
from src.main.util.data import load_pile, get_data_loader


class TestDataLoading(unittest.TestCase):
    """
    Test data loading from util/data
    """
    def test_load_pile(self):
        """
        Check loading pile dataset returns some samples with text for train, test, and val
        :return:
        """
        train, val, test = load_pile()
        for dataset in [train, val, test]:
            i = 10  # Check first 10 samples
            for sample in dataset:
                self.assertIsInstance(sample, str)
                if i == 0:
                    break
                i -= 1

    def test_get_data_loader(self):
        """
        Check get_data_loader functions by checking batch size and samples in a batch
        :return:
        """
        train, val, test = get_data_loader(16)
        for dataloader in [train, val, test]:
            count = 0  # Check size of batch
            batch = next(iter(dataloader))
            for sample in batch:
                self.assertIsInstance(sample, str)
                count += 1
            self.assertEqual(count, 16)


if __name__ == '__main__':
    unittest.main()
