import argparse
import os
from typing import List, Tuple
from pathlib import Path

import time
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adagrad
import numpy as np

from transformers import TrainingArguments, Trainer
import evaluate

from src.main.util import get_data_loader, process_data_to_txt, load_pile
from src.main.inference import load
from src.main.llama import Transformer, Tokenizer, ModelArgs

from datasets import load_dataset
from transformers import LlamaTokenizerFast
from torch.optim import AdamW
from transformers import get_scheduler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metric = evaluate.load("accuracy")
base_path: str = os.path.dirname(__file__).removesuffix(os.path.normpath("/src/main/train"))


def load_model(
        ckpt_dir: str,
        max_seq_len: int,
        max_batch_size: int,
        vocab_size: int
) -> Tuple[Transformer, Tokenizer]:
    start_time = time.time()
    # checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # assert world_size == len(
    #     checkpoints
    # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    # ckpt_path = checkpoints[local_rank]
    print("Loading LLaMa model")
    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    # with open(Path(ckpt_dir) / "params.json", "r") as f:
    #     params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size  # , **params
    )
    model_args.vocab_size = vocab_size
    torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    # model.load_state_dict(checkpoint, strict=False)

    print(f"Loaded model in {time.time() - start_time:.2f} seconds")
    return model


def compute_metrics(eval_prediction):
    print(eval_prediction)
    predictions = np.argmax(eval_prediction[0], axis=-1)
    return metric.compute(predictions=predictions, references=eval_prediction[1])


"""
# Try using Yelp reviews from HuggingFace example
dataset = load_dataset("yelp_review_full")

tokenizer = LlamaTokenizerFast.from_pretrained("output/")  # load from pretrained tokenizer weights
tokenizer.pad_token = -100

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, max_length=32, truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.filter(lambda ex : len(ex["input_ids"]) == 32)

# Convert to data needed for training loop
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
"""


def main(*args, **kwargs):
    seq_len = 64
    batch_size = 2
    lr = 5e-7

    tokenizer = Tokenizer("cl100k_base")
    model = load_model(None, seq_len, batch_size, tokenizer.vocab_size)

    model.to(device)

    # Manual training - code based on HuggingFace example
    small_train_dataset, small_eval_dataset, _ = load_pile(num_train=3000, num_val=100, seq_len=seq_len)

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)

    epochs = 3
    total_train_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_train_steps
    )


    p_bar = tqdm(range(total_train_steps))

    model.train()
    for epoch in range(epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            p_bar.update(1)

    """
    # Train model using HuggingFace Trainer
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset, #train_set,
        eval_dataset=small_eval_dataset, #val_set,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    """


def run_main():
    arg_parser = argparse.ArgumentParser()
    args, _ = arg_parser.parse_known_args()  # Only parse known args
    main(**vars(args))


if __name__ == "__main__":
    run_main()