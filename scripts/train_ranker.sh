#!/bin/bash

python style_ranker/ranker/train.py \
  --referenced_file_path data/ranker_data/mix_code_open/gpt_35/instruction_response/referenced.json \
  --direct_file_path data/ranker_data/mix_code_open/gpt_35/instruction_response/direct.json \
  --human_file_path data/ranker_data/mix_code_open/gpt_35/instruction_response/human.json \
  --referenced_correct_score_path data/ranker_data/mix_code_open/gpt_35/quality_measure/referenced_correct_score.json \
  --direct_correct_score_path data/ranker_data/mix_code_open/gpt_35/quality_measure/direct_correct_score.json \
  --human_correct_score_path data/ranker_data/mix_code_open/gpt_35/quality_measure/human_correct_score.json \
  --referenced_help_score_path data/ranker_data/mix_code_open/gpt_35/quality_measure/referenced_help_score.json \
  --direct_help_score_path data/ranker_data/mix_code_open/gpt_35/quality_measure/direct_help_score.json \
  --human_help_score_path data/ranker_data/mix_code_open/gpt_35/quality_measure/human_help_score.json \
  --hidden_dim 768 \
  --linear_dim 512 \
  --encoder_name Alibaba-NLP/gte-base-en-v1.5 \
  --model_path gte_base_ranker_test \
  --num_epochs 20
