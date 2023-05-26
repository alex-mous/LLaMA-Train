"""
Test default transformer model
"""

import unittest
from src.main.llama.old_model import ModelArgs, Transformer
from src.main.llama.tokenizer import Tokenizer


class TestTransformer(unittest.TestCase):
    """
    Test model in llama/model.
    """
    def test_basic_model(self):
        """

        """
        args = ModelArgs
        model = Transformer(args)


if __name__ == '__main__':
    unittest.main()
