"""
Test input tokenization
"""

import os
import unittest
from src.main.llama.tokenizer import Tokenizer


tokenizer_path: str = os.path.join(
    os.path.dirname(__file__).removesuffix(os.path.normpath("src/test/llama")),
    os.path.normpath("artifacts/base_100000.model")
)


class TestTokenizer(unittest.TestCase):
    """
    Test tokenization from llama/tokenizer
    """
    def test_basic_tokenizer(self):
        """
        Check basic input tokenization.
        """
        tokenizer = Tokenizer(tokenizer_path)
        self.assertTrue(len(tokenizer.encode("in", False, False)) == 1)
        self.assertTrue(len(tokenizer.encode("in", True, False)) == 2)
        self.assertTrue(len(tokenizer.encode("in", True, True)) == 3)


if __name__ == '__main__':
    unittest.main()
