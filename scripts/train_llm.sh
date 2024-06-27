#!/bin/bash

export WANDB_API_KEY=your_wandb_api_key

accelerate launch --main_process_port 29052 --num_processes 2 style_ranker/llm/train.py \
    --output_dir="your_output_dir" \
    --dataset_name="data/llm_sft_data/open/olmo/selection/10000.json" \
    --cache_dir="your_huggingface_cache_dir" \
    --num_train_epochs=3 \
    --logging_steps=1 \
    --save_steps=25000 \
    --model_name="meta-llama/Meta-Llama-3-8B" \
    --save_total_limit=100 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --group_by_length=False \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_ratio=0.03 \
    --weight_decay=0. \
    --optim="paged_adamw_32bit" \
    --bf16=True \
    --tf32=True \
    --remove_unused_columns=False \
    --model_max_length=2048 \
    --run_name="your_process_name" \
    --seed=100 \
    --deepspeed="your_deepspeed_config" \
    --use_peft \

