# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
import argparse
import gc
import time

import torch
import torch.distributed

from src.main.llama import ModelArgs, Tokenizer, XFormersLLaMa, XFormersTransformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(
        tokenizer_path: str,
        checkpoint_path: str,
        **model_args
) -> XFormersLLaMa:
    start_time = time.time()
    print("Loading LLaMa model and tokenizer.")
    # load tokenizer.
    tokenizer = Tokenizer(model_path=tokenizer_path)
    # load model.
    model_params = ModelArgs(**model_args)
    model_params.vocab_size = tokenizer.n_words
    model = XFormersTransformer(model_params)
    model.load_state_dict(torch.load(checkpoint_path))
    # build XFormersLLaMa model.
    torch.set_default_tensor_type(torch.FloatTensor)
    print(f"Loaded model and tokenizer in {time.time() - start_time:.2f} seconds")
    inference_model = XFormersLLaMa(model, tokenizer, device)
    return inference_model


def main(
        model_path: str,
        tokenizer_path: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_gen_len: int = 512
):
    generator = load(
        tokenizer_path,
        model_path,
        dim=512,
        n_layers=8,
        n_heads=8
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
    ]
    results = generator.generate(
        prompts, max_gen_len, temperature, top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")


def run_main():
    arg_parser = argparse.ArgumentParser()
    # paths.
    arg_parser.add_argument(
        "-m",
        "--model-path",
        help="Path to model artifact",
        required=True
    )
    arg_parser.add_argument(
        "-t",
        "--tokenizer-path",
        help="Path to tokenizer artifact"
    )
    # generation settings.
    arg_parser.add_argument("-t", "--temperature", help="Temperature probability", default=0.8, type=float)
    arg_parser.add_argument("-tp", "--top-p", help="Top p probability", default=0.95, type=float)
    arg_parser.add_argument("-l", "--max-gen-len", help="Maximum output length", default=512, type=int)
    # model args.
    arg_parser.add_argument("-d", "--dim", help="Transformer dimension", default=512, type=int)
    arg_parser.add_argument("-nl", "--n-layers", help="Number of transformer layers", default=8, type=int)
    arg_parser.add_argument("-nh", "--n-heads", help="Number of attention heads", default=8, type=int)
    arg_parser.add_argument("-v", "--vocab-size", help="Number of possible words", default=-1, type=int)
    arg_parser.add_argument("-mo", "--multiple-of", help="SwiGLU hidden layer multiple", default=256, type=int)
    arg_parser.add_argument("ne", "--norm-eps", help="Smoothing value for RMSNorm", default=1e-5, type=float)
    args, _ = arg_parser.parse_known_args()
    main(**vars(args))


if __name__ == "__main__":
    try:
        run_main()
    finally:
        gc.collect()
        torch.cuda.empty_cache()
