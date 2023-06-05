import os
import time
import torch
from torch import nn
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
