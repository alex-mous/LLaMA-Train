"""
Test default transformer model
"""

import unittest

import torch

from src.main.llama.old_model import ModelArgs, Transformer
from src.main.llama.tokenizer import Tokenizer


class TestTransformer(unittest.TestCase):
    """
    Test model in llama/model.
    """
    def test_old_model(self):
        """
        Tests basic functionality (constructors and processing several examples) in old_model
        """
        args = ModelArgs
        tokenizer = Tokenizer()
        args.vocab_size = tokenizer.n_words
        model = Transformer(args)
        inputs = ["the cat sat on the mat", "the cats sit on the mats", "the cat sats on the mat"]
        input_tokens = [tokenizer.encode(input) for input in inputs]
        min_prompt_size = min(len(t) for t in input_tokens)  # min token length
        max_prompt_size = max(len(t) for t in input_tokens)
        total_len = 25   # generation lengths
        tokens = torch.full((len(inputs), total_len), -1).long()  # batch size x max possible len
        for k, t in enumerate(input_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()  # fill in prompts
        input_text_mask = tokens != -1  # mask out padding
        start_pos = min_prompt_size
        prev_pos = 0
        # iterate over first non-generated token in batch to predict next token for all prompts in batch
        for cur_pos in range(start_pos, total_len):
            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            next_token = torch.argmax(logits, dim=-1).reshape(-1)

            # replace only if we don't already have this token in our promt
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        # detokenize resulting generations to string
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(input_tokens[i]) + total_len]
            decoded.append(tokenizer.decode(t))
        print(decoded)


if __name__ == '__main__':
    unittest.main()
