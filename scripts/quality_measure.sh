#!/bin/bash
# measure the quality of the generated responses to the instructions
python style_ranker/ranker/quality.py \
  --api_key your_openai_api_key \
  --data_dir data/ranker_data/mix_code_open/gpt_35/instruction_response \
  --output_dir data/ranker_data/mix_code_open/gpt_35/quality_measure \
  --file_list direct.json human.json referenced.json \
  --aspects helpfulness correctness \
  --model gpt-3.5-turbo \
  --batch_size 50