import gc
import os

import torch
from torch.nn import CrossEntropyLoss

from src.main.util import evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_path: str = os.path.join(
    os.path.dirname(__file__).removesuffix(os.path.normpath("src/main/util")),
    os.path.normpath("output/")
)


def main(*args, **kwargs):
    chkpt_base = output_path + "checkpoints/" + "dim-256-heads-8-layers-8-big-run/"
    chkpt_name = "chkpt-1-end-light.pt"

    assert os.path.isfile(chkpt_base + chkpt_name), "Initial checkpoint required to eval"

    # Load model and dataloaders
    model, tokenizer, _, val_dataloader = load_model_and_data(
        output_path,
        tokenizer_path="tokenizer.model",
        train_path="tiny_train.jsonl",
        val_path="tiny_val.jsonl",
        num_train=0,
        num_val=10000,
        batch_size=64,
        dim=256,
        n_layers=8,
        n_heads=8,
        initial_chkpt=chkpt_base + chkpt_name,
        new_chkpt_format=True
    )

    # Perform evaluation
    model.to(device)
    model.eval()
    loss_fn = CrossEntropyLoss(ignore_index=tokenizer.eos_id)
    print(f"Validation loss on model: {evaluate(model, val_dataloader, loss_fn)}")
    model.cpu()


if __name__ == "__main__":
    try:
        main()
    finally:
        print("Cleaning up memory...")
        gc.collect()
        torch.cuda.empty_cache()