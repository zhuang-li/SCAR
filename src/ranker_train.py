import argparse
import json
import torch
from torch import nn
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_processing import load_and_process_data
from dataset import ContrastiveDataset
from model import ClassifierWithEncoder
from utils import filter_ins_ans_pairs

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a model.")
    parser.add_argument("--encoder_name", type=str, default="Salesforce/codet5p-110m-embedding", help="Encoder model name")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension size")
    parser.add_argument("--linear_dim", type=int, default=256, help="Linear layer dimension size")
    parser.add_argument("--output_dim", type=int, default=1, help="Output dimension size")
    parser.add_argument("--score_type", type=str, default="help_correct", choices=["correct", "help", "help_correct"], help="Score type")
    parser.add_argument("--disentangle", action="store_true", help="Whether to disentangle embeddings")
    parser.add_argument("--referenced", action="store_true", help="Whether to use referenced data")
    parser.add_argument("--constraint_mode", type=str, default="abs", choices=["abs", "abs_pos", "diff", "abs_diff", "none"], help="Quality constraint mode")
    parser.add_argument("--constraint_min", type=float, default=1.0, help="Constraint mode minimum value")
    parser.add_argument("--constraint_max", type=float, default=5.0, help="Constraint mode maximum value")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--test_batch_size", type=int, default=32, help="Test batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--display_interval", type=int, default=100, help="Display interval")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to save the best model")
    parser.add_argument("--referenced_file_path", type=str, required=True, help="Path to referenced file")
    parser.add_argument("--direct_file_path", type=str, required=True, help="Path to direct file")
    parser.add_argument("--human_file_path", type=str, required=True, help="Path to human file")
    parser.add_argument("--referenced_correct_score_path", type=str, required=True, help="Path to referenced correct score file")
    parser.add_argument("--direct_correct_score_path", type=str, required=True, help="Path to direct correct score file")
    parser.add_argument("--human_correct_score_path", type=str, required=True, help="Path to human correct score file")
    parser.add_argument("--referenced_help_score_path", type=str, required=True, help="Path to referenced help score file")
    parser.add_argument("--direct_help_score_path", type=str, required=True, help="Path to direct help score file")
    parser.add_argument("--human_help_score_path", type=str, required=True, help="Path to human help score file")
    return parser.parse_args()

def main(args):
    score_type = args.score_type
    disentangle = args.disentangle
    referenced = args.referenced
    constraint_mode = args.constraint_mode
    constraint_min = args.constraint_min
    constraint_max = args.constraint_max

    # Load data
    referenced_correct_scores = json.load(open(args.referenced_correct_score_path, 'r'))
    direct_correct_scores = json.load(open(args.direct_correct_score_path, 'r'))
    human_correct_scores = json.load(open(args.human_correct_score_path, 'r'))

    referenced_help_scores = json.load(open(args.referenced_help_score_path, 'r'))
    direct_help_scores = json.load(open(args.direct_help_score_path, 'r'))
    human_help_scores = json.load(open(args.human_help_score_path, 'r'))

    if score_type == 'correct':
        referenced_scores = referenced_correct_scores
        direct_scores = direct_correct_scores
        human_scores = human_correct_scores
    elif score_type == 'help':
        referenced_scores = referenced_help_scores
        direct_scores = direct_help_scores
        human_scores = human_help_scores
    elif score_type == 'help_correct':
        referenced_scores = {k: (v + referenced_help_scores[k]) / 2 for k, v in referenced_correct_scores.items()}
        direct_scores = {k: (v + direct_help_scores[k]) / 2 for k, v in direct_correct_scores.items()}
        human_scores = {k: (v + human_help_scores[k]) / 2 for k, v in human_correct_scores.items()}

    instructions, human_outputs = load_and_process_data(args.human_file_path)
    referenced_instructions, referenced_outputs = load_and_process_data(args.referenced_file_path)
    direct_instructions, direct_outputs = load_and_process_data(args.direct_file_path)

    for idx, item in enumerate(instructions):
        assert item == direct_instructions[idx]

    for idx, item in enumerate(instructions):
        assert item == referenced_instructions[idx]

    # Split data into training, validation, and test sets
    train_instr, temp_instr, train_human_ans, temp_human_ans, train_referenced_ans, temp_referenced_ans, train_direct_ans, temp_direct_ans = train_test_split(
        instructions, human_outputs, referenced_outputs, direct_outputs, test_size=0.2, random_state=42)
    val_instr, test_instr, val_human_ans, test_human_ans, val_referenced_ans, test_referenced_ans, val_direct_ans, test_direct_ans = train_test_split(
        temp_instr, temp_human_ans, temp_referenced_ans, temp_direct_ans, test_size=0.5, random_state=42)

    # Datasets and DataLoader
    train_dataset = ContrastiveDataset(train_instr, train_direct_ans, train_referenced_ans, train_human_ans)
    val_dataset = ContrastiveDataset(val_instr, val_direct_ans, val_referenced_ans, val_human_ans)
    test_dataset = ContrastiveDataset(test_instr, test_direct_ans, test_referenced_ans, test_human_ans)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # Model initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = ClassifierWithEncoder(args.encoder_name, args.hidden_dim, args.linear_dim, args.output_dim,device=device).to(device)

    # Loss function and optimizer
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * 0.1, num_training_steps=total_steps)

    best_val_accuracy = -float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        classifier.train()
        print(f'Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()}')

        total_loss = 0
        total_direct_referenced_creativity_similarity = 0
        total_direct_referenced_presentation_similarity = 0
        total_referenced_human_creativity_similarity = 0
        total_referenced_human_presentation_similarity = 0
        total_direct_human_creativity_similarity = 0
        total_direct_human_presentation_similarity = 0

        iteration = 0

        for instruction, direct_ans, referenced_ans, human_ans in tqdm(train_loader):
            iteration += 1

            direct_score, direct_creativity_style_embedding, direct_presentation_style_embedding = classifier(
                instruction, direct_ans)
            if referenced:
                referenced_score, referenced_creativity_style_embedding, referenced_presentation_style_embedding = classifier(instruction, referenced_ans)

            human_score, human_creativity_style_embedding, human_presentation_style_embedding = classifier(instruction, human_ans)
            target = torch.ones(human_score.size()).to(device)

            if referenced:
                direct_referenced_indexes = filter_ins_ans_pairs(instruction, direct_ans, referenced_ans, direct_scores, referenced_scores, constraint_mode, min=constraint_min, max=constraint_max)
                referenced_human_indexes = filter_ins_ans_pairs(instruction, referenced_ans, human_ans, referenced_scores, human_scores, constraint_mode, min=constraint_min, max=constraint_max)

            direct_human_indexes = filter_ins_ans_pairs(instruction, direct_ans, human_ans, direct_scores, human_scores, constraint_mode, min=constraint_min, max=constraint_max)

            batch_loss = 0
            if referenced:
                if direct_referenced_indexes:
                    direct_referenced_ranking_criterion = nn.MarginRankingLoss(margin=0.5)
                    loss1 = direct_referenced_ranking_criterion(direct_score[direct_referenced_indexes], referenced_score[direct_referenced_indexes], target[direct_referenced_indexes])
                    total_direct_referenced_creativity_similarity += nn.CosineSimilarity(dim=1)(
                        direct_creativity_style_embedding[direct_referenced_indexes], referenced_creativity_style_embedding[direct_referenced_indexes]).mean().item()

                    total_direct_referenced_presentation_similarity += nn.CosineSimilarity(dim=1)(
                        direct_presentation_style_embedding[direct_referenced_indexes], referenced_presentation_style_embedding[direct_referenced_indexes]).mean().item()
                    batch_loss += loss1

                if referenced_human_indexes:
                    referenced_human_ranking_criterion = nn.MarginRankingLoss(margin=0.5)
                    loss2 = referenced_human_ranking_criterion(referenced_score[referenced_human_indexes], human_score[referenced_human_indexes], target[referenced_human_indexes])
                    total_referenced_human_creativity_similarity += nn.CosineSimilarity(dim=1)(
                        referenced_creativity_style_embedding[referenced_human_indexes], human_creativity_style_embedding[referenced_human_indexes]).mean().item()

                    total_referenced_human_presentation_similarity += nn.CosineSimilarity(dim=1)(
                        referenced_presentation_style_embedding[referenced_human_indexes], human_presentation_style_embedding[referenced_human_indexes]).mean().item()
                    batch_loss += loss2

            if direct_human_indexes:
                direct_human_ranking_criterion = nn.MarginRankingLoss(margin=0.5)
                loss3 = direct_human_ranking_criterion(direct_score[direct_human_indexes], human_score[direct_human_indexes], target[direct_human_indexes])
                total_direct_human_creativity_similarity += nn.CosineSimilarity(dim=1)(
                    direct_creativity_style_embedding[direct_human_indexes], human_creativity_style_embedding[direct_human_indexes]).mean().item()

                total_direct_human_presentation_similarity += nn.CosineSimilarity(dim=1)(
                    direct_presentation_style_embedding[direct_human_indexes], human_presentation_style_embedding[direct_human_indexes]).mean().item()
                batch_loss += loss3

            if referenced:
                common_indexes = list(set(direct_referenced_indexes) & set(referenced_human_indexes) & set(direct_human_indexes))
            else:
                common_indexes = list(set(direct_human_indexes))

            if common_indexes and disentangle:
                creativity_style_loss = TripletMarginLoss(margin=1.0, p=2)(referenced_creativity_style_embedding[common_indexes],
                                                                            human_creativity_style_embedding[common_indexes],
                                                                            direct_creativity_style_embedding[common_indexes])
                presentation_style_loss = TripletMarginLoss(margin=1.0, p=2)(direct_presentation_style_embedding[common_indexes],
                                                                            referenced_presentation_style_embedding[common_indexes],
                                                                            human_presentation_style_embedding[common_indexes])
                loss_weight = 0.1
                batch_loss += loss_weight * ((creativity_style_loss + presentation_style_loss) / 2)

            if batch_loss != 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += batch_loss.item()

            if iteration % args.display_interval == 0:
                avg_loss = total_loss / args.display_interval
                avg_direct_referenced_creativity_similarity = total_direct_referenced_creativity_similarity / args.display_interval
                avg_direct_referenced_presentation_similarity = total_direct_referenced_presentation_similarity / args.display_interval
                avg_referenced_human_creativity_similarity = total_referenced_human_creativity_similarity / args.display_interval
                avg_referenced_human_presentation_similarity = total_referenced_human_presentation_similarity / args.display_interval
                avg_direct_human_creativity_similarity = total_direct_human_creativity_similarity / args.display_interval
                avg_direct_human_presentation_similarity = total_direct_human_presentation_similarity / args.display_interval

                print(f'Epoch {epoch + 1}, Iteration {iteration}')
                print(f'Average Loss: {avg_loss:.4f}')
                print(f'Average Direct Referenced Creativity Similarity: {avg_direct_referenced_creativity_similarity:.4f}')
                print(f'Average Direct Referenced Presentation Similarity: {avg_direct_referenced_presentation_similarity:.4f}')
                print(f'Average Referenced Human Creativity Similarity: {avg_referenced_human_creativity_similarity:.4f}')
                print(f'Average Referenced Human Presentation Similarity: {avg_referenced_human_presentation_similarity:.4f}')
                print(f'Average Direct Human Creativity Similarity: {avg_direct_human_creativity_similarity:.4f}')
                print(f'Average Direct Human Presentation Similarity: {avg_direct_human_presentation_similarity:.4f}')

                total_loss = 0
                total_direct_referenced_creativity_similarity = 0
                total_direct_referenced_presentation_similarity = 0
                total_referenced_human_creativity_similarity = 0
                total_referenced_human_presentation_similarity = 0
                total_direct_human_creativity_similarity = 0
                total_direct_human_presentation_similarity = 0

        classifier.eval()
        correct1 = 0
        correct2 = 0
        correct3 = 0
        gap1 = 0
        gap2 = 0
        gap3 = 0
        total = 0
        with torch.no_grad():
            for instruction, direct_ans, referenced_ans, human_ans in tqdm(val_loader):
                direct_score, _, _ = classifier(instruction, direct_ans)
                referenced_score, _, _ = classifier(instruction, referenced_ans)
                human_score, _, _ = classifier(instruction, human_ans)
                correct1 += (direct_score > referenced_score).sum().item()
                correct2 += (direct_score > human_score).sum().item()
                correct3 += (referenced_score > human_score).sum().item()
                gap1 += (direct_score - referenced_score).sum().item()
                gap2 += (direct_score - human_score).sum().item()
                gap3 += (referenced_score - human_score).sum().item()
                total += direct_score.size(0)

        val_accuracy = (correct1 + correct2 + correct3) / (total * 3)
        print(f'Epoch {epoch + 1}, Val Acc: {val_accuracy}')
        val_gap = (gap1 + gap2 + gap3) / (total * 3)
        print(f'Epoch {epoch + 1}, Val Gap: {val_gap}')

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = classifier.state_dict().copy()
            print(f'Saving best model with validation accuracy: {best_val_accuracy:.4f}')

    print(f"Save to path: {args.model_path}")
    torch.save(best_model_state, args.model_path)

    def evaluate_model(model, data_loader):
        model.eval()
        correct1 = 0
        correct2 = 0
        correct3 = 0
        correct4 = 0
        total = 0
        total_direct_referenced_creativity_similarity = 0
        total_direct_referenced_presentation_similarity = 0
        total_referenced_human_creativity_similarity = 0
        total_referenced_human_presentation_similarity = 0
        total_direct_human_creativity_similarity = 0
        total_direct_human_presentation_similarity = 0
        display_interval = 0
        with torch.no_grad():
            for instruction, direct_ans, referenced_ans, human_ans in tqdm(data_loader):
                display_interval += 1
                direct_score, direct_creativity_style_embedding, direct_presentation_style_embedding = classifier(
                    instruction, direct_ans)
                referenced_score, referenced_creativity_style_embedding, referenced_presentation_style_embedding = classifier(instruction, referenced_ans)
                human_score, human_creativity_style_embedding, human_presentation_style_embedding = classifier(instruction, human_ans)
                correct1 += (direct_score > referenced_score).sum().item()
                correct2 += (direct_score > human_score).sum().item()
                correct3 += (referenced_score > human_score).sum().item()
                condition1 = torch.gt(direct_score, referenced_score)
                condition2 = torch.gt(referenced_score, human_score)
                combined_condition = torch.logical_and(condition1, condition2)
                correct4 += combined_condition.sum().item()
                total += direct_score.size(0)
                total_direct_referenced_creativity_similarity += nn.CosineSimilarity(dim=1)(
                    direct_creativity_style_embedding, referenced_creativity_style_embedding).mean().item()

                total_direct_referenced_presentation_similarity += nn.CosineSimilarity(dim=1)(
                    direct_presentation_style_embedding, referenced_presentation_style_embedding).mean().item()

                total_referenced_human_creativity_similarity += nn.CosineSimilarity(dim=1)(
                    referenced_creativity_style_embedding, human_creativity_style_embedding).mean().item()

                total_referenced_human_presentation_similarity += nn.CosineSimilarity(dim=1)(
                    referenced_presentation_style_embedding, human_presentation_style_embedding).mean().item()

                total_direct_human_creativity_similarity += nn.CosineSimilarity(dim=1)(
                    direct_creativity_style_embedding, human_creativity_style_embedding).mean().item()

                total_direct_human_presentation_similarity += nn.CosineSimilarity(dim=1)(
                    direct_presentation_style_embedding, human_presentation_style_embedding).mean().item()

        accuracy = (correct1 + correct2 + correct3) / (total * 3)
        avg_direct_referenced_creativity_similarity = total_direct_referenced_creativity_similarity / display_interval
        avg_direct_referenced_presentation_similarity = total_direct_referenced_presentation_similarity / display_interval
        avg_referenced_human_creativity_similarity = total_referenced_human_creativity_similarity / display_interval
        avg_referenced_human_presentation_similarity = total_referenced_human_presentation_similarity / display_interval
        avg_direct_human_creativity_similarity = total_direct_human_creativity_similarity / display_interval
        avg_direct_human_presentation_similarity = total_direct_human_presentation_similarity / display_interval

        print(f'Average Direct Referenced Creativity Similarity: {avg_direct_referenced_creativity_similarity:.4f}')
        print(f'Average Direct Referenced Presentation Similarity: {avg_direct_referenced_presentation_similarity:.4f}')
        print(f'Average Referenced Human Creativity Similarity: {avg_referenced_human_creativity_similarity:.4f}')
        print(f'Average Referenced Human Presentation Similarity: {avg_referenced_human_presentation_similarity:.4f}')
        print(f'Average Direct Human Creativity Similarity: {avg_direct_human_creativity_similarity:.4f}')
        print(f'Average Direct Human Presentation Similarity: {avg_direct_human_presentation_similarity:.4f}')
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"direct_score > referenced_score accuracy: {correct1 / total:.4f}")
        print(f"direct_score > human_score accuracy: {correct2 / total:.4f}")
        print(f"referenced_score > human_score accuracy: {correct3 / total:.4f}")
        print(f"direct_score > referenced_score > human_score accuracy: {correct4 / total:.4f}")

    classifier.load_state_dict(best_model_state)
    evaluate_model(classifier, test_loader)

if __name__ == "__main__":
    args = parse_args()
    main(args)
