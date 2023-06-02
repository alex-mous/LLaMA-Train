"""
Test default transformer model
"""

import unittest

import torch

from src.main.llama.model import ModelArgs, Transformer
from src.main.llama.tokenizer import Tokenizer


class TestTransformer(unittest.TestCase):
    """
    Test model in llama/model.
    """
    def test_model(self):
        """
        Tests basic functionality (constructors and processing several examples) in old_model
        """
        args = ModelArgs()
        tokenizer = Tokenizer()
        args.vocab_size = tokenizer.vocab_size
        model = Transformer(args)
        input_tokens = torch.tensor([
            [2, 64, 92, 108],
            [9801, 20, 94, 5],
            [20, 91, 890, 1555]
        ]).long()
        outputs = model(input_tokens)
        self.assertTrue(outputs.shape == (input_tokens.shape[0], input_tokens.shape[1], tokenizer.vocab_size))


if __name__ == '__main__':
    unittest.main()
