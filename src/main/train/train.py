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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metric = evaluate.load("accuracy")
base_path: str = os.path.dirname(__file__).removesuffix(os.path.normpath("/src/main/train"))


def load_model(
        ckpt_dir: str,
        local_rank: int,
        world_size: int,
        max_seq_len: int,
        max_batch_size: int,
) -> Tuple[Transformer, Tokenizer]:
    start_time = time.time()
    #checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    #assert world_size == len(
    #    checkpoints
    #), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    #ckpt_path = checkpoints[local_rank]
    print("Loading")
    #checkpoint = torch.load(ckpt_path, map_location="cpu")
    # with open(Path(ckpt_dir) / "params.json", "r") as f:
    #     params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size  # , **params
    )
    tokenizer = Tokenizer("cl100k")
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    # model.load_state_dict(checkpoint, strict=False)

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer


def compute_metrics(eval_prediction):
    predictions = np.argmax(eval_prediction[0], axis=-1)
    return metric.compute(predictions=predictions, references=eval_prediction[1])


def train(
        model: Transformer,
        tokenizer: Tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        weight_decay: float,
        epochs: int
) -> List[float]:
    # Untested
    pass
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        for raw_text in tqdm(train_loader, position=0, leave=True):
            raw_tokens = tokenizer.encode(raw_text)
            min_prompt_size = min(len(t) for t in raw_tokens)  # min token length
            max_prompt_size = max(len(t) for t in raw_tokens)
            total_len = 25  # generation lengths
            tokens = torch.full((len(raw_text), total_len), -1).long()  # batch size x max possible len
            for k, t in enumerate(raw_tokens):
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
            optimizer.zero_grad()

            # Todo: predict single token at some point for each sample in batch, and compare to true token
            pred = model(raw_text)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        print(f"Epoch {epoch + 1}. "
              f"Train Loss={round(train_losses[-1], 4)}. "
              f"Total Time={round((time.time() - epoch_start_time) / 60, 2)}m")
    print(f"Total training time={round((time.time() - start_time) / 60, 2)}m")
    return train_losses



def main(*args, **kwargs):
    # TODO: setup data
    train_set, val_set, test_set = load_pile()

    # TODO: load model based on checkpoints, params from example code
    model, tokenizer = load_model(None, -1, -1, 512, 32)

    print(f"Training model")

    # Train model using HuggingFace Trainer
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
    )
    trainer.train()


def preprocess_data():
    # Process datasets to text files and train tokenizer
    artifacts_path = os.path.join(base_path, os.path.join("data"))
    train_text_path = os.path.join(artifacts_path, "07_medium.txt")
    if os.path.exists(train_text_path):
        print(f"File \"{train_text_path}\" is already loaded.")
    else:
        train_data, val_data, test_data = load_pile()
        process_data_to_txt(train_data, train_text_path, p=1e-2)


def run_main():
    arg_parser = argparse.ArgumentParser()
    args, _ = arg_parser.parse_known_args()  # Only parse known args
    main(**vars(args))


if __name__ == "__main__":
    preprocess_data()
    # run_main()
