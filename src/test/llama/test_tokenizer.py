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
        tokenizer = Tokenizer()
        self.assertTrue(len(tokenizer.encode("in")) == 1)
        self.assertTrue(len(tokenizer.encode("the")) == 1)
        self.assertEqual("hello world", tokenizer.decode(tokenizer.encode("hello world")))


if __name__ == '__main__':
    unittest.main()
