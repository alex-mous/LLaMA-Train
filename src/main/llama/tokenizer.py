"""
SentencePieceProcessor-based Tokenizer, based off of original LLaMa tokenizer

This code is based off of code from Meta Platforms, Inc., and unmodified portions are attributed to the following declaration:
Copyright (c) Meta Platforms, Inc. and affiliates.
This software may be used and distributed according to the terms of the GNU General Public License version 3.
"""

from typing import List
from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, model_path: str):
        # Load tokenizer from tokenizer model
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # Copy special tokens from model
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
