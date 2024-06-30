# SCAR: Style Consistency-Aware Response Ranking for LLM Instruction-Tuning

## Overview

SCAR is an innovative data selection method that enhances instruction-tuning for large language models. By leveraging style consistency-aware response ranking, SCAR identifies and selects the most beneficial training data for fine-tuning LLMs, ultimately improving their performance.

## Installation

Ensure you have a **Python 3.8+** environment. You can install SCAR using pip:

```bash
pip install scar-tool
```

## Requirements

SCAR requires the following dependencies: `torch>=2.3`, `transformers>=4.37`, `huggingface_hub>=0.23`, `scikit-learn`, `tqdm`, `nltk` and `datasketch`. These will be automatically installed when you install SCAR via pip.


## Usage

### Basic Usage with Hugging Face Transformers

Here's a simple example of how to use the StyleRanker model with Hugging Face Transformers:

```python
import torch
from transformers import AutoTokenizer
from style_ranker.ranker.model import StyleRanker

# Load the model and tokenizer
model_path = "lizhuang144/scar-gte-base"
model = StyleRanker.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare your data
instructions = ["Write a poem about spring", "Explain quantum computing"]
answers = ["Blossoms bloom in gentle breeze...", "Quantum computing is a type of computation..."]

# Tokenize the inputs
max_length = 512
instruction_inputs = tokenizer(instructions, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
answer_inputs = tokenizer(answers, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
model.eval()
# Get the scores
with torch.no_grad():
    scores = model(
        instruction_inputs.input_ids,
        instruction_inputs.attention_mask,
        answer_inputs.input_ids,
        answer_inputs.attention_mask
    )

# Print the results
for instruction, answer, score in zip(instructions, answers, scores):
    print(f"Instruction: {instruction}")
    print(f"Answer: {answer}")
    print(f"Score: {score.item()}")
    print()
```

### Advanced Usage

SCAR offers sophisticated capabilities for data filtering and ranking through its comprehensive pipeline. This allows
you to fine-tune your selection process by choosing the top-k pairs with the highest scores, setting a ratio for
selection, or applying a threshold for filtering.

The `rank_and_filter` function provides a powerful way to rank and filter instruction-answer pairs. Here's an example
demonstrating its usage:

```python
from style_ranker.rank import rank_and_filter
import torch
# Load the model and tokenizer
model_path = "lizhuang144/scar-gte-base"

# Prepare your data
instructions = ["Write a poem about spring", "Explain quantum computing", "Describe the water cycle"]
answers = ["Blossoms bloom in gentle breeze...", "Quantum computing is a type of computation...",
           "The water cycle, also known as..."]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Example 1: Using topk
topk_pairs = rank_and_filter(model_path, instructions, answers, topk=2, device=device)

# Example 2: Using threshold
threshold_pairs = rank_and_filter(model_path, instructions, answers, threshold=-0.5, device=device)

# Example 3: Using ratio
ratio_pairs = rank_and_filter(model_path, instructions, answers, ratio=0.5, device=device)

# Print results for each method
print("Top-k results:")
for instruction, answer, score in topk_pairs:
    print(f"Score: {score:.2f} | Instruction: {instruction}")

print("\nThreshold results:")
for instruction, answer, score in threshold_pairs:
    print(f"Score: {score:.2f} | Instruction: {instruction}")

print("\nRatio results:")
for instruction, answer, score in ratio_pairs:
    print(f"Score: {score:.2f} | Instruction: {instruction}")
```
## Model List

We provide the following pre-trained SCAR models:

- [`lizhuang144/scar-gte-base`](https://huggingface.co/lizhuang144/scar-gte-base): SCAR model trained using [`Alibaba-NLP/gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) as the representation encoder.
- [`lizhuang144/scar-gte-large`](https://huggingface.co/lizhuang144/scar-gte-large): SCAR model trained using [`Alibaba-NLP/gte-large-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) as the representation encoder.
- [`lizhuang144/scar-roberta-base`](https://huggingface.co/lizhuang144/scar-roberta-base): SCAR model trained using [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base) as the representation encoder.
## Performance

SCAR demonstrates significant improvements in LLM performance when used for data filtering and selection. We evaluated
our method using two LLMs: Olmo and Starcoder.

**Note:** Prior to applying SCAR, we filter out non-English and remove duplicate instruction-response pairs.

### Olmo Performance

| Dataset Size        | L.C. WinRate |
|---------------------|--------------|
| Full dataset (320k) | 3.86         |
| SCAR-filtered 10k   | 5.37         |
| SCAR-filtered 5k    | 5.64         |
| SCAR-filtered 2.5k  | 4.08         |

The official checkpoint [allenai/OLMo-7B-SFT](https://huggingface.co/allenai/OLMo-7B-SFT) is trained on 320k data from [allenai/tulu-v2-sft-mixture](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture). We evaluate the performance of models trained with SCAR-filtered data using 10k, 5k, and 2.5k instruction-answer pairs. The evaluation metric is L.C. WinRate, which compares model outputs with 'gpt-4-1106-preview' using [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) as the judger on the [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) benchmark.


### Starcoder Performance

| Dataset Size       | HumanEval (Python)                                 | MultiPL-E (Java)                                 | MultiPL-E (JavaScript)                                  | MultiPL-E (C++)                |
|--------------------|----------------------------------------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------|
|                    | Pass@1 / Pass@10                                   | Pass@1 / Pass@10                                 | Pass@1 / Pass@10                                 | Pass@1 / Pass@10               |
| Full dataset (13k) | 35.56/ 51.81 | 26.03    / 38.44             | 32.80   / 46.97      | 29.32                  / 41.90 |
| SCAR-filtered 10k  | 36.29 / 53.99    | 28.29      / 39.58       | 33.22     / 49.79   | 30.17       / 46.20            |
| SCAR-filtered 5k   | 36.95    / 54.07    | 28.96     / 39.02    | 34.53  / 49.90 | 34.53      / 49.90             |
| SCAR-filtered 2.5k | 37.57 / 55.65 | 29.29   / 41.06 | 34.09  / 49.47 | 31.19   / 42.83                |

The official checkpoint ['bigcode/octocoder'](https://huggingface.co/bigcode/octocoder) is the ['bigcode/starcoder'](https://huggingface.co/bigcode/starcoder) fine-tuned on 13k data from ['bigcode/guanaco-commits'](https://huggingface.co/datasets/bigcode/guanaco-commits). We evaluated the performance using the [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness). The performance of 'bigcode/octocoder' is obtained from the ['bigcode/bigcode-models-leaderboard'](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard/tree/main/community_results/bigcode_octocoder_loubnabnl/metrics_octocoder). We evaluated models on four datasets in four programming languages (Python, Java, C++, and JavaScript) and reported two execution accuracies (Pass@1 and Pass@10) for each dataset. We evaluated the performance of the model trained with SCAR-filtered data with 10k, 5k, and 2.5k instruction-answer pairs.


## Key Components

- **StyleRanker**: A model for ranking instruction-answer pairs based on style consistency and data quality.
- **Data Filtering**: Scripts for filtering and selecting high-quality instruction-answer pairs.
- **LLM Training**: Scripts for fine-tuning large language models using the selected data.

## Scripts

The `scripts/` directory contains bash scripts for various tasks:

- `data_synthesize.sh`: Synthesizes 'referenced' and 'direct' responses based on the human responses for training the ranker. Please adjust the script arguments as needed.
- `quality_measure.sh`: Measures the quality of the collected responses using LLMs, utilized to train the ranker.
- `train_ranker.sh`: Trains the SCAR style ranker model. Please update the script arguments as needed.
- `data_filter.sh`: Ranks and filters instruction-answer pairs. Please update the script arguments as needed.
- `train_llm.sh`: This script fine-tunes a large language model using the filtered data. Review and update the script arguments accordingly to ensure proper training. 
The following additional packages are required to train the LLM: `peft`, `trl`, `accelerate` and `deepspeed`.

Ensure all dependencies are installed before running these scripts to achieve the best results.

## Project Structure

The project is organized as follows:

- `data/`: Datasets for training and evaluation
    - `llm_sft_data/`: Training data for the large language model (code and open domain)
    - `ranker_data/`: Training data for the ranker (code and open domain)
- `style_ranker/`: Main package
    - `consts.py`
    - `dedup.py`: Near deduplication
    - `llm/`: LLM training (`train.py`)
    - `rank.py`: Ranking and filtering
    - `ranker/`: StyleRanker implementation
        - `config.py`, `dataset.py`, `model.py`, `quality.py`: Quality measure with LLMs like GPT-3.5-turbo
        - SCAR ranker training (`train.py`)
    - `utils.py`
- `examples/`: Example Python scripts
    - `filter_pipeline.py`, `rank_pairs.py`, `remove_dupes.py`, `vicuna_converter.py`
- `scripts/`: Example Bash scripts
    - `data_filter.sh`, `quality_measure.sh`, `train_llm.sh`, `train_ranker.sh`
- `requirements.txt`: List of dependencies
- `setup.py`: Installation script

