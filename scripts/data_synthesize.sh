#!/bin/bash

python style_ranker/ranker/data_synthesize.py \
  --api_key your_openai_api_key \
  --input_file data/ranker_data/mix_code_open/gpt_35/instruction_response/human.json \
  --output_file data/ranker_data/mix_code_open/gpt_35/instruction_response/referenced.json \
  --prompt_template referenced \
  --model gpt-3.5-turbo \
  --batch_size 100
