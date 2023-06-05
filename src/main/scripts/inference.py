"""
Run inference on a model checkpoint
"""
import argparse
import gc

import torch
import torch.distributed

from src.main.llama import XFormersLLaMa, load_llama

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(
        model_path: str,
        tokenizer_path: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_gen_len: int = 512,
        **model_args
):
    """
    Run generation on a model with given params
    """
    model, tokenizer = load_llama(
        tokenizer_path,
        model_path,
        new_chkpt_type=True,
        **model_args
    )
    generator = XFormersLLaMa(model, tokenizer, device)

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "The meaning of life is ",
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
    """
    Parse args and run inference
    """
    arg_parser = argparse.ArgumentParser()
    # paths.
    arg_parser.add_argument(
        "-mp",
        "--model-path",
        help="Path to model checkpoint",
        required=True
    )
    arg_parser.add_argument(
        "-tp",
        "--tokenizer-path",
        help="Path to tokenizer model",
        required=True
    )
    # generation settings.
    arg_parser.add_argument(
        "-t",
        "--temperature",
        help="Temperature probability",
        default=0.8,
        type=float
    )
    arg_parser.add_argument(
        "-p",
        "--top-p",
        help="Top p probability",
        default=0.95,
        type=float
    )
    arg_parser.add_argument(
        "-gl",
        "--max-gen-len",
        help="Maximum output length",
        default=512,
        type=int
    )
    # model args.
    arg_parser.add_argument(
        "-d",
        "--dim",
        help="Transformer dimension",
        default=512,
        type=int
    )
    arg_parser.add_argument(
        "-nl",
        "--n-layers",
        help="Number of transformer layers",
        default=8,
        type=int
    )
    arg_parser.add_argument(
        "-nh",
        "--n-heads",
        help="Number of attention heads",
        default=8,
        type=int
    )
    arg_parser.add_argument(
        "-mo",
        "--multiple-of",
        help="SwiGLU hidden layer multiple",
        default=256,
        type=int
    )
    arg_parser.add_argument(
        "-ne",
        "--norm-eps",
        help="Smoothing value for RMSNorm",
        default=1e-5,
        type=float
    )
    args, _ = arg_parser.parse_known_args()
    main(**vars(args))


if __name__ == "__main__":
    try:
        run_main()
    finally:
        gc.collect()
        torch.cuda.empty_cache()
