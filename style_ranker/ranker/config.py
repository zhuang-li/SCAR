import json
import os

from transformers import PretrainedConfig, AutoConfig


class StyleRankerConfig(PretrainedConfig):
    model_type = "style_ranker"

    def __init__(
            self,
            encoder_name="FacebookAI/roberta-base",
            hidden_dim=1024,
            linear_dim=1024,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_name = encoder_name
        self.hidden_dim = hidden_dim
        self.linear_dim = linear_dim

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Check if this is a path to a saved config
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_file):
            # Load the saved config
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
        else:
            # If no config file found, try to download it
            try:
                print(f"Attempting to download config from {pretrained_model_name_or_path}")
                auto_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
                config_dict = auto_config.to_dict()
            except Exception as e:
                print(f"Failed to download config: {e}")
                # If download fails, use the provided kwargs
                config_dict = kwargs
                config_dict['encoder_name'] = pretrained_model_name_or_path

        # Update with any provided kwargs, allowing for overrides
        config_dict.update(kwargs)

        return cls(**config_dict)

AutoConfig.register("style_ranker", StyleRankerConfig)