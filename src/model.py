import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from utils import max_pooling


class ClassifierWithEncoder(nn.Module):
    def __init__(self, encoder_name, hidden_dim, linear_dim, output_dim, device='cuda'):
        super(ClassifierWithEncoder, self).__init__()
        self.device = device
        self.encoder_name = encoder_name
        if 'codet5p' in encoder_name:
            self.code_t5_encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)
            self.encoder = self.code_t5_encoder.encoder
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)

        self.relation_network = nn.Sequential(
            nn.Linear(linear_dim * 2, linear_dim),
            nn.ReLU(),
            nn.Linear(linear_dim, hidden_dim)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, instruction, answer):
        instruction_input = self.tokenizer(instruction, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        answer_input = self.tokenizer(answer, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)

        instruction_output = self.encoder(**instruction_input)
        answer_output = self.encoder(**answer_input)

        instruction_output_last_hidden_state = instruction_output.last_hidden_state
        answer_output_last_hidden_state = answer_output.last_hidden_state

        if 'codet5p' in self.encoder_name:
            composition_style_embedding = self.relation_network(torch.cat(
                (F.normalize(self.code_t5_encoder.proj(instruction_output_last_hidden_state[:, 0, :]), dim=-1),
                 F.normalize(self.code_t5_encoder.proj(answer_output_last_hidden_state[:, 0, :]), dim=-1)), dim=1
            ))
        else:
            composition_style_embedding = self.relation_network(torch.cat(
                (instruction_output.pooler_output, answer_output.pooler_output), dim=1
            ))

        answer_attention_mask = answer_input['attention_mask']
        presentation_style_embedding = max_pooling(answer_output.last_hidden_state, answer_attention_mask)
        combined_output = torch.cat((composition_style_embedding, presentation_style_embedding), dim=1)
        x = self.output_layer(combined_output)
        return x, composition_style_embedding, presentation_style_embedding
