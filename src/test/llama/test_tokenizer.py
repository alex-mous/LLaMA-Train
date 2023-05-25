"""
Test input tokenization
"""

import unittest
from src.main.llama.tokenizer import Tokenizer


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

    def test_tokenizer_vocab_size(self):
        tokenizer = Tokenizer()
        self.assertEqual(100277, tokenizer.n_words)


if __name__ == '__main__':
    unittest.main()
