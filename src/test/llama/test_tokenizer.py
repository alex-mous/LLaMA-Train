"""
Test input tokenization
"""

import os
import unittest
from src.main.llama.tokenizer import Tokenizer
from src.main.util.data import load_pile, get_data_loader


tokenizer_path: str = os.path.join(
    os.path.dirname(__file__).removesuffix(os.path.normpath("src/test/llama")),
    os.path.normpath("artifacts/")
)


class TestTokenizer(unittest.TestCase):
    """
    Test tokenization from llama/tokenizer
    """
    def test_load_pile(self):
        """
        Check basic input tokenization.
        """
        # tokenizer = Tokenizer(tokenizer_path)
        pass


if __name__ == '__main__':
    unittest.main()
