import json
import numpy as np
import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer
from accelerate import PartialState

from style_ranker.consts import SYSTEM_PROMPT

# Constants
IGNORE_TOKEN_ID = -100


# Helper Functions
def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}")


def create_datasets(tokenizer, args, seed=None):
    with open(args.dataset_name, "r") as f:
        raw_data = json.load(f)

    np.random.seed(seed)
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * 0.995)
    train_indices, eval_indices = perm[:split], perm[split:]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]

    print(f"Size of the train set: {len(train_raw_data)}. Size of the validation set: {len(eval_raw_data)}")

    train_dataset = SupervisedDataset(args, train_raw_data, tokenizer, formatting_func=preprocess)
    valid_dataset = SupervisedDataset(args, eval_raw_data, tokenizer, formatting_func=preprocess)
    return train_dataset, valid_dataset


# Data Processing
def preprocess(args, sources, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    conversations = [
        SYSTEM_PROMPT.format(PROBLEM=source[0]['value'], OUTPUT=source[1]['value'], EOS_TOKEN=tokenizer.eos_token) for
        source in sources]

    return_token_type_ids = 'OLMo' not in args.model_name

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_token_type_ids=return_token_type_ids,
    ).input_ids

    targets = input_ids.clone()

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            total_len += 1

        rounds = conversation.split(tokenizer.eos_token)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID

        for rou in rounds:
            if not rou:
                break

            parts = rou.split(" ASSISTANT: ")
            if len(parts) != 2:
                break

            parts[0] += " ASSISTANT: "
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += round_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_TOKEN_ID
            print(
                f'Error Processing Attention!!! cur_len: {cur_len} total_len: {total_len} tokenizer.model_max_length: {tokenizer.model_max_length}')

    return {
        "input_ids": input_ids,
        "labels": targets,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id),
    }


# Dataset Class
class SupervisedDataset(Dataset):
    def __init__(self, args, raw_data, tokenizer: transformers.PreTrainedTokenizer, formatting_func=None):
        super().__init__()
        sources = [example["conversations"] for example in raw_data]
        data_dict = formatting_func(args, sources, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "attention_mask": self.attention_mask[i],
        }


# Script Arguments
@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B", metadata={"help": "the model name"})
    cache_dir: Optional[str] = field(default="hugginface/cache_dir", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="data/llm_sft_data/open/olmo/selection/10000.json", metadata={"help": "the dataset name"})
    model_max_length: Optional[int] = field(default=2048, metadata={"help": "model_max_length"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "the target modules for LoRA"},
    )
    use_auth_token: Optional[str] = field(default=None, metadata={"help": "the Hugging Face auth token"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "whether to use PEFT"})


# Main Execution
def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    if script_args.use_peft:
        print("Using LoRA")
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=script_args.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        print("Not using LoRA")
        peft_config = None

    if training_args.group_by_length and script_args.packing:
        raise ValueError("Cannot use both packing and group by length")

    if training_args.gradient_checkpointing:
        raise ValueError("gradient_checkpointing not supported")

    set_seed(training_args.seed)

    device_string = PartialState().process_index
    print('-' * 50)
    print(torch.__version__)

    use_fast = 'OLMo' in script_args.model_name

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        cache_dir=script_args.cache_dir,
        use_auth_token=script_args.use_auth_token,
        trust_remote_code=True,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=use_fast,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset, eval_dataset = create_datasets(tokenizer, script_args, seed=training_args.seed)

    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        cache_dir=script_args.cache_dir,
        device_map={"": device_string},
        trust_remote_code=True,
        use_auth_token=script_args.use_auth_token,
    )
    base_model.config.use_cache = False

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=script_args.packing,
        max_seq_length=script_args.model_max_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()