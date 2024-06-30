#!/bin/bash

python examples/filter_pipeline.py \
  --model_path lizhuang144/scar-gte-base \
  --input_file allenai/tulu-v2-sft-mixture \
  --output_file data/llm_sft_data/open/olmo/advanced_selection/10000_alpaca_format.json \
  --topk 10000 \
  --vicuna_output data/llm_sft_data/open/olmo/advanced_selection/10000.json \
  --skip_deduplication