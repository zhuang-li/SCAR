import argparse
import json
import torch
from torch import nn
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from style_ranker.ranker.dataset import ContrastiveDataset
from style_ranker.ranker.model import StyleRanker, StyleRankerConfig
from style_ranker.utils import load_and_process_data, filter_ins_ans_pairs


class StyleRankerTrainer:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Use from_pretrained to create the config
        self.config = StyleRankerConfig(
            args.encoder_name,
            hidden_dim=args.hidden_dim,
            linear_dim=args.linear_dim,
        )
        self.max_length = args.max_text_length
        self.model = StyleRanker(self.config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, trust_remote_code=True)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        self.ranking_criterion = nn.MarginRankingLoss(margin=0.5)
        self.triplet_criterion = TripletMarginLoss(margin=1.0, p=2)
        self.disentangle_loss_weight = args.disentangle_loss_weight

    def load_data(self):
        score_files = {
            'referenced': {'correct': self.args.referenced_correct_score_path,
                           'help': self.args.referenced_help_score_path},
            'direct': {'correct': self.args.direct_correct_score_path, 'help': self.args.direct_help_score_path},
            'human': {'correct': self.args.human_correct_score_path, 'help': self.args.human_help_score_path}
        }

        scores = {}
        for source in score_files:
            if self.args.score_type == 'help_correct':
                correct_scores = json.load(open(score_files[source]['correct'], 'r'))
                help_scores = json.load(open(score_files[source]['help'], 'r'))
                scores[source] = {k: (correct_scores[k] + help_scores[k]) / 2 for k in correct_scores}
            else:
                scores[source] = json.load(open(score_files[source][self.args.score_type], 'r'))

        instructions, human_outputs = load_and_process_data(self.args.human_file_path)
        referenced_instructions, referenced_outputs = load_and_process_data(self.args.referenced_file_path)
        direct_instructions, direct_outputs = load_and_process_data(self.args.direct_file_path)

        assert all(
            instructions[i] == direct_instructions[i] == referenced_instructions[i] for i in range(len(instructions)))

        return instructions, human_outputs, referenced_outputs, direct_outputs, scores

    def prepare_data_loaders(self, instructions, human_outputs, referenced_outputs, direct_outputs):
        train_data, val_test_data = train_test_split(
            list(zip(instructions, human_outputs, referenced_outputs, direct_outputs)), test_size=0.2, random_state=42)
        test_data, val_data = train_test_split(val_test_data, test_size=0.5, random_state=42)  # 0.25 x 0.8 = 0.2

        def create_dataset(data):
            return ContrastiveDataset(*zip(*data))

        train_loader = DataLoader(create_dataset(train_data), batch_size=self.args.train_batch_size, shuffle=True)
        val_loader = DataLoader(create_dataset(val_data), batch_size=self.args.val_batch_size, shuffle=False)
        test_loader = DataLoader(create_dataset(test_data), batch_size=self.args.test_batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader, scheduler, scores):
        self.model.train()
        total_loss = 0
        total_batches = 0

        pbar = tqdm(train_loader, desc="Training")

        for iteration, (instruction, human_ans, referenced_ans, direct_ans) in enumerate(pbar):

            instruction_inputs = self.tokenizer(instruction, return_tensors='pt', padding=True, truncation=True,
                                                max_length=self.max_length).to(self.device)
            direct_inputs = self.tokenizer(direct_ans, return_tensors='pt', padding=True, truncation=True,
                                           max_length=self.max_length).to(self.device)
            referenced_inputs = self.tokenizer(referenced_ans, return_tensors='pt', padding=True, truncation=True,
                                               max_length=self.max_length).to(self.device)
            human_inputs = self.tokenizer(human_ans, return_tensors='pt', padding=True, truncation=True,
                                          max_length=self.max_length).to(self.device)

            direct_score, direct_creativity, direct_presentation = self.model(
                instruction_inputs.input_ids, instruction_inputs.attention_mask,
                direct_inputs.input_ids, direct_inputs.attention_mask
            )
            referenced_score, referenced_creativity, referenced_presentation = self.model(
                instruction_inputs.input_ids, instruction_inputs.attention_mask,
                referenced_inputs.input_ids, referenced_inputs.attention_mask
            )
            human_score, human_creativity, human_presentation = self.model(
                instruction_inputs.input_ids, instruction_inputs.attention_mask,
                human_inputs.input_ids, human_inputs.attention_mask
            )

            target = torch.ones(human_score.size()).to(self.device)
            #breakpoint()
            direct_referenced_indexes = filter_ins_ans_pairs(instruction, direct_ans, referenced_ans, scores['direct'],
                                                             scores['referenced'], self.args.constraint_mode,
                                                             min_thres=self.args.constraint_min, max_thres=self.args.constraint_max)
            referenced_human_indexes = filter_ins_ans_pairs(instruction, referenced_ans, human_ans,
                                                            scores['referenced'], scores['human'],
                                                            self.args.constraint_mode, min_thres=self.args.constraint_min,
                                                            max_thres=self.args.constraint_max)
            direct_human_indexes = filter_ins_ans_pairs(instruction, direct_ans, human_ans, scores['direct'],
                                                        scores['human'], self.args.constraint_mode,
                                                        min_thres=self.args.constraint_min, max_thres=self.args.constraint_max)

            batch_loss = 0
            if direct_referenced_indexes:
                batch_loss += self.ranking_criterion(direct_score[direct_referenced_indexes],
                                                     referenced_score[direct_referenced_indexes],
                                                     target[direct_referenced_indexes])
            if referenced_human_indexes:
                batch_loss += self.ranking_criterion(referenced_score[referenced_human_indexes],
                                                     human_score[referenced_human_indexes],
                                                     target[referenced_human_indexes])
            if direct_human_indexes:
                batch_loss += self.ranking_criterion(direct_score[direct_human_indexes],
                                                     human_score[direct_human_indexes],
                                                     target[direct_human_indexes])

            if self.args.disentangle:
                common_indexes = list(
                    set(direct_referenced_indexes) & set(referenced_human_indexes) & set(direct_human_indexes))
                if common_indexes:
                    creativity_loss = self.triplet_criterion(referenced_creativity[common_indexes],
                                                             human_creativity[common_indexes],
                                                             direct_creativity[common_indexes])
                    presentation_loss = self.triplet_criterion(direct_presentation[common_indexes],
                                                               referenced_presentation[common_indexes],
                                                               human_presentation[common_indexes])
                    batch_loss += self.disentangle_loss_weight * (creativity_loss + presentation_loss) / 2


            if batch_loss != 0:
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                scheduler.step()
                total_loss += batch_loss.item()
                total_batches += 1

            if (iteration + 1) % self.args.display_interval == 0:
                #breakpoint()
                avg_loss = total_loss / total_batches if total_batches > 0 else 0
                pbar.set_postfix({'Avg Loss': f'{avg_loss:.4f}'})

        epoch_loss = total_loss / total_batches if total_batches > 0 else 0
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        correct = [0, 0, 0]
        total = 0
        with torch.no_grad():
            for instruction, human_ans, referenced_ans, direct_ans in tqdm(val_loader):
                instruction_inputs = self.tokenizer(instruction, return_tensors='pt', padding=True, truncation=True,
                                                    max_length=self.max_length).to(self.device)
                direct_inputs = self.tokenizer(direct_ans, return_tensors='pt', padding=True, truncation=True,
                                               max_length=self.max_length).to(self.device)
                referenced_inputs = self.tokenizer(referenced_ans, return_tensors='pt', padding=True, truncation=True,
                                                   max_length=self.max_length).to(self.device)
                human_inputs = self.tokenizer(human_ans, return_tensors='pt', padding=True, truncation=True,
                                              max_length=self.max_length).to(self.device)

                direct_score = self.model(instruction_inputs.input_ids, instruction_inputs.attention_mask,
                                          direct_inputs.input_ids, direct_inputs.attention_mask)
                referenced_score = self.model(instruction_inputs.input_ids, instruction_inputs.attention_mask,
                                              referenced_inputs.input_ids, referenced_inputs.attention_mask)
                human_score = self.model(instruction_inputs.input_ids, instruction_inputs.attention_mask,
                                         human_inputs.input_ids, human_inputs.attention_mask)

                correct[0] += (direct_score > referenced_score).sum().item()
                correct[1] += (direct_score > human_score).sum().item()
                correct[2] += (referenced_score > human_score).sum().item()
                total += direct_score.size(0)

        accuracy = sum(correct) / (total * 3)
        return accuracy

    def train(self):
        instructions, human_outputs, referenced_outputs, direct_outputs, scores = self.load_data()
        train_loader, val_loader, test_loader = self.prepare_data_loaders(instructions, human_outputs,
                                                                          referenced_outputs, direct_outputs)

        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=len(train_loader) * self.args.num_epochs * 0.1,
                                                    num_training_steps=len(train_loader) * self.args.num_epochs)

        best_val_accuracy = -float('inf')
        for epoch in range(self.args.num_epochs):
            train_loss = self.train_epoch(train_loader, scheduler, scores)
            val_accuracy = self.validate(val_loader)

            print(f'Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.model.save_pretrained(self.args.model_path)
                self.tokenizer.save_pretrained(self.args.model_path)
                print(f'Saving best model with validation accuracy: {best_val_accuracy:.4f}')

        self.model = StyleRanker.from_pretrained(self.args.model_path).to(self.device)
        test_accuracy = self.validate(test_loader)
        print(f'Test Accuracy: {test_accuracy:.4f}')


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a StyleRanker model.")
    parser.add_argument("--encoder_name", type=str, default="FacebookAI/roberta-base",
                        help="Encoder model name")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension size")
    parser.add_argument("--linear_dim", type=int, default=1024, help="Linear layer dimension size")
    parser.add_argument("--score_type", type=str, default="help_correct", choices=["correct", "help", "help_correct"],
                        help="Score type")
    parser.add_argument("--disentangle", action="store_true",default=True, help="Whether to disentangle embeddings")
    parser.add_argument("--constraint_mode", type=str, default="abs",
                        choices=["abs", "abs_pos", "diff", "abs_diff", "none"], help="Quality constraint mode")
    parser.add_argument("--constraint_min", type=float, default=5.0, help="Constraint mode minimum value")
    parser.add_argument("--constraint_max", type=float, default=10.0, help="Constraint mode maximum value")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--test_batch_size", type=int, default=32, help="Test batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--display_interval", type=int, default=100, help="Display interval")
    parser.add_argument("--model_path", type=str, default="best_model", help="Directory path to save the best model")
    parser.add_argument("--referenced_file_path", type=str, required=True, help="Path to referenced file")
    parser.add_argument("--direct_file_path", type=str, required=True, help="Path to direct file")
    parser.add_argument("--human_file_path", type=str, required=True, help="Path to human file")
    parser.add_argument("--referenced_correct_score_path", type=str, required=True,
                        help="Path to referenced correct score file")
    parser.add_argument("--direct_correct_score_path", type=str, required=True,
                        help="Path to direct correct score file")
    parser.add_argument("--human_correct_score_path", type=str, required=True, help="Path to human correct score file")
    parser.add_argument("--referenced_help_score_path", type=str, required=True,
                        help="Path to referenced help score file")
    parser.add_argument("--direct_help_score_path", type=str, required=True, help="Path to direct help score file")
    parser.add_argument("--human_help_score_path", type=str, required=True, help="Path to human help score file")
    parser.add_argument("--disentangle_loss_weight", type=float, default=0.1, help="Weight for disentanglement loss")
    parser.add_argument("--max_text_length", type=int, default=512, help="Maximum text length for input documents")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = StyleRankerTrainer(args)
    trainer.train()