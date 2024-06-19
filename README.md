# SCAR: Efficient Instruction-Tuning for Large Language Models via Style Consistency-Aware Response Ranking

## Overview
SCAR is a data selection method designed to enhance instruction-tuning for large language models by leveraging style consistency-aware response ranking.

## Project Structure
The project is organized into several key directories:

- `data`: Contains all datasets for training and evaluation.
  - `llm_sft_data`: Data for training the large language model (LLM).
    - `code`: Code-related datasets.
    - `open`: Open datasets.
  - `ranker_data`: Data for training the ranker.
    - `code`: Code-related datasets for the ranker.
    - `open`: Open datasets for the ranker.
- `src`: Contains all source code files for the project.
- `requirements.txt`: Lists all the dependencies required for the project.

## Installation

### Prerequisites
Before you begin, ensure you have the following dependencies installed:

- PyTorch
- Transformers
- tqdm
- scikit-learn

You can install these dependencies using conda:

```sh
conda install pytorch torchvision torchaudio -c pytorch
conda install transformers
conda install tqdm
conda install scikit-learn
```

## Data

### SFT Data (`llm_sft_data`)
This directory contains data used for training the large language model. It is divided into subdirectories based on the data source (e.g., `code`, `open`). Each subdirectory contains various JSON files organized by different selection criteria.

### Ranker Data (`ranker_data`)
This directory contains data used for training the ranker. It is divided into subdirectories based on the data source (e.g., `code`, `open`). Each subdirectory contains:
- `instruction_response`: JSON files with instruction-response pairs.
  - `direct.json`: Direct responses to the instructions.
  - `human.json`: Human-generated responses.
  - `referenced.json`: Referenced responses to the instructions.
- `quality_measure`: JSON files with quality scores for the responses.
  - `direct_correct_score.json`: Quality scores for the correctness of direct responses.
  - `direct_help_score.json`: Quality scores for the helpfulness of direct responses.
  - `human_correct_score.json`: Quality scores for the correctness of human responses.
  - `human_help_score.json`: Quality scores for the helpfulness of human responses.
  - `referenced_correct_score.json`: Quality scores for the correctness of referenced responses.
  - `referenced_help_score.json`: Quality scores for the helpfulness of referenced responses.

## Running the Project

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/SCAR.git
   cd SCAR/src
    ```

2. **Prepare the data**
   Ensure your data files are placed in the respective directories (`data/llm_sft_data/` and `data/ranker_data/`).

3. **Train and Evaluate the Ranker**
   Execute the `ranker_train.py` script with appropriate arguments. Example:
   ```sh
   python ranker_train.py --encoder_name Salesforce/codet5p-110m-embedding \
                          --hidden_dim 768 \
                          --linear_dim 256 \
                          --output_dim 1 \
                          --score_type help_correct \
                          --disentangle \
                          --referenced \
                          --constraint_mode abs \
                          --train_batch_size 4 \
                          --val_batch_size 16 \
                          --test_batch_size 32 \
                          --learning_rate 5e-5 \
                          --num_epochs 10 \
                          --display_interval 100 \
                          --model_path best_model.pth \
                          --referenced_file_path ../data/ranker_data/code/gpt_35/instruction_response/referenced.json \
                          --direct_file_path ../data/ranker_data/code/gpt_35/instruction_response/direct.json \
                          --human_file_path ../data/ranker_data/code/gpt_35/instruction_response/human.json \
                          --referenced_correct_score_path ../data/ranker_data/code/gpt_35/quality_measure/referenced_correct_score.json \
                          --direct_correct_score_path ../data/ranker_data/code/gpt_35/quality_measure/direct_correct_score.json \
                          --human_correct_score_path ../data/ranker_data/code/gpt_35/quality_measure/human_correct_score.json \
                          --referenced_help_score_path ../data/ranker_data/code/gpt_35/quality_measure/referenced_help_score.json \
                          --direct_help_score_path ../data/ranker_data/code/gpt_35/quality_measure/direct_help_score.json \
                          --human_help_score_path ../data/ranker_data/code/gpt_35/quality_measure/human_help_score.json

