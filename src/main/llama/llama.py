import time
import torch

from typing import Tuple, Optional
from src.main.llama import Transformer, Tokenizer, ModelArgs


def load_llama(
        tokenizer_path: str,
        checkpoint_dir: Optional[str],
        initial_checkpoint: Optional[str],
        **model_args
) -> Tuple[Transformer, Tokenizer]:
    # Load LLaMa model and tokenizer with given parameters
    # TODO: checkpoint loading
    start_time = time.time()
    print("Loading LLaMa model and tokenizer")
    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_params = ModelArgs(**model_args)
    model_params.vocab_size = tokenizer.n_words
    model = Transformer(model_params)
    torch.set_default_tensor_type(torch.FloatTensor)
    print(f"Loaded model and tokenizer in {time.time() - start_time:.2f} seconds")
    return model, tokenizer
