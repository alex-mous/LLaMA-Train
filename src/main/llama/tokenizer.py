from typing import List

import tiktoken


class Tokenizer:
    def __init__(self, model: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(model)
        self.n_words = self.tokenizer.n_vocab

    def encode(self, s: str) -> List[int]:
        return self.tokenizer.encode(s)

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)
