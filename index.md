---
layout: default
title: SCAR: Data Selection via Style Consistency Aware Response Ranking
description: SCAR selects high quality instruction–answer pairs by style consistency to enable efficient instruction tuning of large language models.
---

# [ACL 2025 main] SCAR: Data Selection via Style Consistency Aware Response Ranking for Efficient Instruction Tuning of Large Language Models

## Overview

SCAR is an innovative data-selection method that enhances instruction tuning for large language models. By leveraging style-consistency-aware response ranking, SCAR identifies and selects the most beneficial training data for instruction tuning, ultimately improving model performance.

<p align="center">
  <a href="https://arxiv.org/abs/2406.10882"><img src="https://img.shields.io/badge/arXiv-2406.10882-b31b1b.svg" alt="arXiv"></a>
  <a href="https://pypi.org/project/scar-tool/"><img src="https://img.shields.io/pypi/v/scar-tool?color=g" alt="PyPI"></a>
  <a href="https://pepy.tech/project/scar-tool"><img src="https://static.pepy.tech/badge/scar-tool" alt="Downloads"></a>
  <a href="https://github.com/zhuang-li/SCAR"><img src="https://img.shields.io/github/stars/zhuang-li/SCAR?style=social" alt="GitHub stars"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT license"></a>
</p>

## Installation

Ensure you have **Python 3.8+** installed, then run:

```bash
pip install scar-tool
````

## Requirements

`torch >= 2.3`, `transformers >= 4.37`, `huggingface_hub >= 0.23`, `scikit-learn`, `tqdm`, `nltk`, `datasketch`.
These packages install automatically with **scar-tool**.

## Usage

### Basic example (Hugging Face Transformers)

```python
import torch
from transformers import AutoTokenizer
from style_ranker.ranker.model import StyleRanker

model_path = "lizhuang144/scar-gte-base"
model = StyleRanker.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

instructions = ["Write a poem about spring",
                "Explain quantum computing"]
answers = ["I am sorry. Who are you? Why should I tell you anything about poem",
           "Quantum computing is a type of computation..."]

ins = tokenizer(instructions, return_tensors="pt", padding=True,
                truncation=True, max_length=512)
ans = tokenizer(answers, return_tensors="pt", padding=True,
                truncation=True, max_length=512)

model.eval()
with torch.no_grad():
    scores = model(ins.input_ids, ins.attention_mask,
                   ans.input_ids, ans.attention_mask)

for i, (instr, answ, s) in enumerate(zip(instructions, answers, scores)):
    print(f"{i+1}. {instr}\n   {answ}\n   Score: {s.item():.2f}\n")
```

### Advanced: rank and filter

```python
from style_ranker.rank import rank_and_filter
import torch

model_path = "lizhuang144/scar-gte-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

instructions = ["Write a poem about spring",
                "Explain quantum computing",
                "Describe the water cycle"]
answers = ["I am sorry. Who are you? Why should I tell you anything about poem",
           "Quantum computing is a type of computation...",
           "The water cycle, also known as..."]

topk_pairs = rank_and_filter(model_path, instructions, answers, topk=2, device=device)
threshold_pairs = rank_and_filter(model_path, instructions, answers, threshold=-2.0, device=device)
ratio_pairs = rank_and_filter(model_path, instructions, answers, ratio=0.5, device=device)
```

> **Tip:** SCAR models currently do **not** support non-English data or automatic de-duplication. Exclude non-English examples and remove duplicates before filtering.

## Model List

We provide the following pre-trained SCAR models:

- [`lizhuang144/scar-gte-base`](https://huggingface.co/lizhuang144/scar-gte-base): SCAR model trained using [`Alibaba-NLP/gte-base-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) as the representation encoder.
- [`lizhuang144/scar-gte-large`](https://huggingface.co/lizhuang144/scar-gte-large): SCAR model trained using [`Alibaba-NLP/gte-large-en-v1.5`](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) as the representation encoder.
- [`lizhuang144/scar-roberta-base`](https://huggingface.co/lizhuang144/scar-roberta-base): SCAR model trained using [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base) as the representation encoder.

The models here are pre-trained on a dataset consisting primarily of open-domain data, available at `data/ranker_data/mix_code_open/gpt_35`.

## Performance

### Olmo (AlpacaEval, L.C. WinRate ↑)

| Data size  | WinRate  |
| ---------- | -------- |
| Full 320 k | 3.86     |
| SCAR 10 k  | **5.37** |
| SCAR 5 k   | **5.64** |
| SCAR 2.5 k | 4.08     |

### Starcoder (Pass\@1 / Pass\@10)

| Data size  | Python            | Java              | JavaScript        | C++               |
| ---------- | ----------------- | ----------------- | ----------------- | ----------------- |
| Full 13 k  | 35.56 / 51.81     | 26.03 / 38.44     | 32.80 / 46.97     | 29.32 / 41.90     |
| SCAR 10 k  | 36.29 / 53.99     | 28.29 / 39.58     | 33.22 / 49.79     | 30.17 / 46.20     |
| SCAR 5 k   | **36.95 / 54.07** | **28.96 / 39.02** | **34.53 / 49.90** | **32.83 / 44.47** |
| SCAR 2.5 k | 37.57 / 55.65     | 29.29 / 41.06     | 34.09 / 49.47     | 31.19 / 42.83     |

## Key components

* **StyleRanker** – ranks instruction–answer pairs by style consistency
* **Data filtering scripts** – select high quality data
* **LLM training scripts** – train models on filtered data

## Scripts

| Script               | Purpose                                                      |
| -------------------- | ------------------------------------------------------------ |
| `data_synthesize.sh` | generate referenced and direct responses for ranker training |
| `quality_measure.sh` | score responses with LLMs                                    |
| `train_ranker.sh`    | train StyleRanker                                            |
| `data_filter.sh`     | rank and filter instruction–answer pairs                     |
| `train_llm.sh`       | train an LLM on filtered data                                |

Extra packages for LLM training: `peft`, `trl`, `accelerate`, `deepspeed`.

## Project structure

```
data/
  llm_sft_data/
  ranker_data/
style_ranker/
  consts.py  dedup.py  llm/  rank.py  ranker/  utils.py
examples/
scripts/
requirements.txt
setup.py
```

## Citation

```bibtex
@article{li2024scar,
  title   = {SCAR: Data Selection via Style Consistency Aware Response Ranking for Efficient Instruction Tuning of Large Language Models},
  author  = {Li, Zhuang and Hua, Yuncheng and Vu, Thuy-Trang and Zhan, Haolan and Qu, Lizhen and Haffari, Gholamreza},
  journal = {arXiv preprint arXiv:2406.10882},
  year    = {2024}
}
```

## License

MIT License – © 2024 Zhuang Li

Unlock efficient instruction tuning for your next LLM project with **SCAR**!

