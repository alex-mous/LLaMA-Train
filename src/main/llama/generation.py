# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .model import XFormersTransformer
from .tokenizer import Tokenizer


class XFormersLLaMa:
    def __init__(
            self,
            model: XFormersTransformer,
            tokenizer: Tokenizer,
            device: torch.device = None
    ):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def generate(
            self,
            prompts: List[str],
            max_gen_len: int = 512,
            temperature: float = 0.8,
            top_p: float = 0.95
    ) -> List[str]:
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        assert max_prompt_size < max_gen_len

        tokens = torch.full((len(prompts), max_gen_len), self.tokenizer.eos_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t).long()
        input_text_mask = (tokens != self.tokenizer.eos_id)

        start_pos = min_prompt_size
        self.model.eval()
        for cur_pos in range(start_pos, max_gen_len):
            logits = self.model(tokens)[:, cur_pos, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def load_model_and_data(
    storage_base_path: str,
    tokenizer_path: str,
    train_path: str,
    val_path: str,
    initial_chkpt: Optional[str] = None,
    num_train: int = 20000,
    num_val: int = 10000,
    max_seq_len: int = 512,
    batch_size: int = 16,
    new_chkpt_format: bool = False,
    **model_args
) -> Tuple[XFormersTransformer, Tokenizer, DataLoader, DataLoader]:
    """
    Load a model and train and val dataloaders for training, evaluation, or generation
    """
    assert os.path.isfile(storage_base_path + tokenizer_path), "LLaMa tokenizer pretrained model file required"
    assert os.path.isfile(storage_base_path + train_path), "Train data subset in JSONL format required"
    assert os.path.isfile(storage_base_path + val_path), "Validation data subset in JSONL format required"

    # Load model
    torch.cuda.empty_cache()
    model, tokenizer = load_llama(
        tokenizer_path=storage_base_path + tokenizer_path,
        initial_chkpt=initial_chkpt,
        use_xformers=True,
        new_chkpt=new_chkpt_format,
        max_seq_len=max_seq_len,
        **model_args
    )

    # Load data
    train_set, val_set, _ = load_pile_dataset(
        tokenizer,
        storage_base_path + train_path,
        storage_base_path + val_path,
        num_train=num_train,
        num_val=num_val,
        max_seq_len=max_seq_len,
    )

    train_dataloader, val_dataloader, _ = get_pile_dataloaders(
        train_set,
        val_set,
        batch_size=batch_size
    )

    return model, tokenizer, train_dataloader, val_dataloader