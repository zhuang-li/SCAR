import os

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.utils import cached_file

from style_ranker.ranker.config import StyleRankerConfig
from style_ranker.utils import max_pooling


class StyleRanker(PreTrainedModel):
    config_class = StyleRankerConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.encoder_name, trust_remote_code=True)

        self.relation_network = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.linear_dim),
            nn.ReLU(),
            nn.Linear(config.linear_dim, config.hidden_dim)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.linear_dim),
            nn.ReLU(),
            nn.Linear(config.linear_dim, 1)
        )

    import torch

    def forward(self, instruction_input_ids, instruction_attention_mask, answer_input_ids, answer_attention_mask):
        instruction_output = self.encoder(input_ids=instruction_input_ids, attention_mask=instruction_attention_mask)
        answer_output = self.encoder(input_ids=answer_input_ids, attention_mask=answer_attention_mask)

        # # Debugging information
        # print("Instruction output keys:", instruction_output.keys())
        # print("Answer output keys:", answer_output.keys())

        # Get instruction embedding
        if hasattr(instruction_output, 'pooler_output') and instruction_output.pooler_output is not None:
            instruction_embedding = instruction_output.pooler_output
        elif hasattr(instruction_output, 'last_hidden_state') and instruction_output.last_hidden_state is not None:
            instruction_embedding = instruction_output.last_hidden_state[:, 0]
        else:
            raise ValueError("Neither pooler_output nor last_hidden_state is available for instruction")

        # Get answer embedding
        if hasattr(answer_output, 'pooler_output') and answer_output.pooler_output is not None:
            answer_embedding = answer_output.pooler_output
        elif hasattr(answer_output, 'last_hidden_state') and answer_output.last_hidden_state is not None:
            answer_embedding = answer_output.last_hidden_state[:, 0]
        else:
            raise ValueError("Neither pooler_output nor last_hidden_state is available for answer")

        # # Debugging information
        # print("Instruction embedding shape:", instruction_embedding.shape)
        # print("Answer embedding shape:", answer_embedding.shape)

        creativity_style_embedding = self.relation_network(torch.cat(
            (instruction_embedding, answer_embedding), dim=1
        ))

        presentation_style_embedding = max_pooling(answer_output.last_hidden_state, answer_attention_mask)
        combined_output = torch.cat((creativity_style_embedding, presentation_style_embedding), dim=1)
        score = self.output_layer(combined_output)

        if self.training:
            return score, creativity_style_embedding, presentation_style_embedding
        else:
            return score

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            print(f"Loading config from {pretrained_model_name_or_path}")
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        model = cls(config)

        # print(config.encoder_name)
        # print(config.encoder_name)

        # Check if the path is a directory (local) or a model name (Hugging Face Hub)
        if os.path.isdir(pretrained_model_name_or_path):
            print(f"Loading model from local directory: {pretrained_model_name_or_path}")
            model_file = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            if not os.path.exists(model_file):
                raise ValueError(f"Model file not found in {pretrained_model_name_or_path}")
            state_dict = torch.load(model_file, map_location="cpu")
        else:
            print(f"Loading model from Hugging Face Hub: {pretrained_model_name_or_path}")
            # Use cached_file from transformers.utils
            try:
                archive_file = cached_file(pretrained_model_name_or_path, "pytorch_model.bin", use_auth_token=kwargs.get("use_auth_token"))
                state_dict = torch.load(archive_file, map_location="cpu")
            except Exception as e:
                raise ValueError(f"Error loading model from {pretrained_model_name_or_path}: {e}")

        # Load the state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")

        return model

    def save_pretrained(self, save_directory, max_shard_size=None, safe_serialization=False):
        super().save_pretrained(save_directory)
        torch.save(self.state_dict(), save_directory + "/pytorch_model.bin")

AutoModel.register(StyleRankerConfig, StyleRanker)