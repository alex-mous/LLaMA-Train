import os
import time
import torch
from torch.utils.data import DataLoader

from typing import Tuple, Optional
from src.main.llama import XFormersTransformer, Tokenizer, ModelArgs
from src.main.util import load_pile_dataset, get_pile_dataloaders


def load_llama(
        tokenizer_path: str,
        initial_chkpt: str = None,
        new_chkpt_type: bool = False,  # New checkpointing method
        **model_args
) -> Tuple[XFormersTransformer, Tokenizer]:
    # Load LLaMa model and tokenizer with given parameters
    start_time = time.time()
    print("Loading LLaMa model and tokenizer")
    tokenizer = Tokenizer(model_path=tokenizer_path)
    if initial_chkpt is not None and not new_chkpt_type:
        print(f"Loading initial checkpoint from {initial_chkpt}")
        model = torch.load(initial_chkpt, map_location="cpu")
    else:
        model_params = ModelArgs(**model_args)
        model_params.vocab_size = tokenizer.n_words
        model = XFormersTransformer(model_params)
        if initial_chkpt is not None:
            model.load_state_dict(torch.load(initial_chkpt))
    torch.set_default_tensor_type(torch.FloatTensor)
    print(f"Loaded model and tokenizer in {time.time() - start_time:.2f} seconds")
    return model, tokenizer


def load_llama_and_data(
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
    tokenizer_full_path = os.path.join(storage_base_path, os.path.normpath(tokenizer_path))
    train_full_path = os.path.join(storage_base_path, os.path.normpath(train_path))
    val_full_path = os.path.join(storage_base_path, os.path.normpath(val_path))
    assert os.path.isfile(tokenizer_full_path), f"LLaMa tokenizer pretrained model file required. {tokenizer_full_path} not valid."
    assert os.path.isfile(train_full_path), f"Train data subset in JSONL format required. {train_full_path} not valid."
    assert os.path.isfile(val_full_path), f"Validation data subset in JSONL format required. {val_full_path} not valid."
    if initial_chkpt is not None:
        assert os.path.isfile(initial_chkpt), f"Initial checkpoint path {initial_chkpt} is not valid."

    # Load model
    torch.cuda.empty_cache()
    model, tokenizer = load_llama(
        tokenizer_path=tokenizer_full_path,
        initial_chkpt=initial_chkpt,
        new_chkpt_type=new_chkpt_format,
        max_seq_len=max_seq_len,
        **model_args
    )

    # Load data
    train_set, val_set, _ = load_pile_dataset(
        tokenizer,
        train_full_path,
        val_full_path,
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
