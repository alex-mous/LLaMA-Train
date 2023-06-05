# LLaMA-Train
Small-scale training code for LLaMa using open-source inference code ([LLaMa Inference](https://github.com/facebookresearch/llama)), using the open-source Pile dataset ([Pile data](https://the-eye.eu/public/AI/pile/)).

## Goals
1. Implement basic training code for LLaMA on a small scale
2. Use a small subset of the Pile dataset to train small LLaMA models to perform ablation studies on different model types

## Requirements
- `torch`
- `sentencepiece`
- `torchvision`
- `tqdm`
- `xformers`
- A copy of the tokenizer model from Meta to tokenize data (see below)
- A small subset of a Pile train subset (see below)
- Pile validaton set (see below)

## Setup
*Note: the following instructions are for using LLaMA-Train on a computer with a GPU. To use the Google Colab notebook supplied under `notebooks`, which provides the same functionality, see the acompanying document.*

First, install the requirements with `pip install -r requirements.txt`.

To train and evaluate the model, download data from [Pile](https://the-eye.eu/public/AI/pile/). We used a subset of Pile subset `07` to train our models, and the first `10k` sequences from `val` to evaluate each model. These files will need to be decompressed to JSONL files and stored under the root directory in a folder titled `data`, and rename the train file to `train.jsonl` and the validation file to `val.jsonl`.

Next, we will also need a copy of the tokenizer model from Meta to tokenize data, which can be requested via the [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform). Store the `tokenizer.model` file under `data` as well.

Finally, setup the local structure by adding the root directory of the project to your Python path. For example, the following command does this on Bash: `export PYTHONPATH="$PWD"`

## Usage
For training, evaluation, and generation, run the corresponding script in the `main/scripts` directory. Each script is configured to parse command line arguments and is set to show a detailed explanation of each parameter when provided with the `--help` argument.

## Layout
- `main/llama` contains the model, tokenizer and model generation code, which is based on [LLaMa Inference](https://github.com/facebookresearch/llama), heavily modified to fit the goals of this project
- `main/util` contains data loading and processing, metric computation (loss calculation), and checkpointing code
- `main/scripts` contains scripts to run training, evaluation, and inference for various model parameters
- `notebooks` contains notebooks used to train models on Google Colab, and uses the same code as available in `src`

## License
This project utilizes large portions of the [LLaMa Inference](https://github.com/facebookresearch/llama) code. See the [License](LICENSE) file.
