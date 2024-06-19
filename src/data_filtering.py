import argparse
import json
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_processing import load_and_process_data
from dataset import TextDataset
from model import ClassifierWithEncoder
from utils import evaluate_model, filter_data

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model and filter data based on scores")
    parser.add_argument('--input_file', type=str, required=True, help="Input data file to be filtered")
    parser.add_argument('--encoder_name', type=str, default='Salesforce/codet5p-110m-embedding', help="Name of the encoder model")
    parser.add_argument('--hidden_dim', type=int, default=768, help="Hidden dimension size")
    parser.add_argument('--linear_dim', type=int, default=256, help="Linear layer dimension size")
    parser.add_argument('--output_dim', type=int, default=1, help="Output dimension size")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help="Device to use for evaluation")
    parser.add_argument('--ratios', type=str, default='50,25,12.5', help="Comma-separated list of ratios for filtering data")
    parser.add_argument('--output_filename', type=str, required=True, help="Output filename for the filtered data")
    return parser.parse_args()

def main():
    args = parse_args()

    instructions, outputs = load_and_process_data(args.input_file)
    labels = [1] * len(instructions)

    dataset = TextDataset(instructions, outputs, labels)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    classifier = ClassifierWithEncoder(args.encoder_name, args.hidden_dim, args.linear_dim, args.output_dim).to(args.device)
    classifier.load_state_dict(torch.load(args.model_path))

    sorted_data = evaluate_model(classifier, data_loader, args.device)
    pickle.dump(sorted_data, open('sorted_data.pkl', 'wb'))

    ratio_list = [float(r) for r in args.ratios.split(',')]
    for ratio in ratio_list:
        output_filename = args.output_filename.replace("{ratio}", str(ratio))
        filter_data(sorted_data, ratio, output_filename)

if __name__ == "__main__":
    main()
