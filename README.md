# LLaMA-Train
Small-scale training code for LLaMa using open-source inference code ([LLaMa Inference](https://github.com/facebookresearch/llama)), using the open-source Pile dataset ([Pile data](https://the-eye.eu/public/AI/pile/)).

## Goals
1. Implement training for LLaMa using the Pile dataset
2. Scale the 7B LLaMa model to feasibly train models on a subset of Pile with low compute
3. Perform an ablation study by modifying the ratio of dimension to number of heads on the aforementioned scaled model

## Requirements
- `torch`
- `fairscale`
- `fire`
- `sentencepiece`
- `torchvision`
- `pillow`
- `tqdm`
- `scikit-learn`

## Setup
First, install the requirements with `pip install -r requirements.txt`.

To train and evaluate the model, download data from [Pile](https://the-eye.eu/public/AI/pile/). We used train subset `07` to train our models, and the first `10k` examples from `val` to evaluate each model.

## Usage
See accompanying document.


## Code Base

### Layout
- `llama` contains the model, tokenizer and model output generation code from [LLaMa Inference](https://github.com/facebookresearch/llama), modified to fit the goals of this project.
- `main/inference` contains code to evaluate the models on `val` and `test`
- `main/train` contains code to train the models on the input train set
- `main/util` contains utility methods defined to simplify training and evaluation

### Modifications to Existing LLaMa Code
- Replace attention modules with PyTorch equivalents
- Scale 7B model to reduce compute required for training
  - Reduce number of layers to `n`
  - Reduce training batch size to `B`
- Parameterize model based on dimension and number of heads to perform ablation

### Data Processing Implementation
See accompanying document.

### Training Implementation
See accompanying document.

## Results
See accompanying document.

## License
This project utilizes large portions of the [LLaMa Inference](https://github.com/facebookresearch/llama) code. See the [License](LICENSE) file.
