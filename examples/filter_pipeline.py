import argparse
import json
import re
from typing import List, Dict, Tuple

import datasets
import torch
from tqdm import tqdm

from style_ranker.utils import convert_to_vicuna_format
from style_ranker.rank import rank_and_filter
from style_ranker.dedup import lsh_near_deduplication, select_distinct_indexes

def calculate_english_char_percentage(text):
    # Count the total number of characters
    total_chars = len(text)

    # Count the number of English characters using a simple regex
    english_chars = len(re.findall(r'[a-zA-Z]', text))

    # Calculate the percentage of English characters
    if total_chars == 0:
        return 0
    return (english_chars / total_chars) * 100

def is_english(text: str) -> bool:
    # simple heuristic to determine if a text is in English
    return True if calculate_english_char_percentage(text) > 70 else False



def convert_olmo_to_json(dataset):
    data_list = []
    for data in dataset:
        messages = data['messages']
        if messages[0]['role'] == 'system':
            messages = messages[1:]

        instruction = messages[0]['content']
        assert messages[0]['role'] == 'user', f"{data}"
        response = messages[1]['content']
        assert messages[1]['role'] == 'assistant', f"{response} {messages[1]['role']}"

        data_list.append({"instruction": instruction,
                          "output": response})

    return data_list

def convert_guanaco_to_json(dataset):
    data_list = []

    for data in dataset:
        instruction = data['prompt'].strip()
        response = data['completion'].strip()

        data_list.append({"instruction": instruction,
                            "output": response})

    return data_list


def load_data(file_path: str) -> List[Dict]:
    if file_path == 'allenai/tulu-v2-sft-mixture':
        olmo_data = datasets.load_dataset(file_path, split='train')
        data_list = convert_olmo_to_json(olmo_data)
        return data_list
    elif file_path == 'bigcode/guanaco-commits':
        octo_data = datasets.load_dataset(file_path, split='train')
        octo_data_test = datasets.load_dataset(file_path, split='test')
        data_list = convert_guanaco_to_json(octo_data) + convert_guanaco_to_json(octo_data_test)
        return data_list
    else:
        with open(file_path, 'r') as f:
            return json.load(f)


def save_data(data: List[Dict], file_path: str):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def filter_english(data: List[Dict]) -> List[Dict]:
    return [item for item in tqdm(data) if is_english(item['instruction']) and is_english(item['output'])]


def deduplicate_data(data_list):
    seen = set()
    unique_data = []

    for item in data_list:
        instruction = item['instruction'].strip()
        response = item['output'].strip()
        identifier = (instruction, response)

        if identifier not in seen:
            unique_data.append(item)
            seen.add(identifier)

    return unique_data


def near_deduplicate_data(data: List[Dict], similarity_threshold: float) -> List[Dict]:
    dedup_texts = [f"{item['instruction']} {item['output']}" for item in data]
    duplicate_clusters = lsh_near_deduplication(dedup_texts, num_perm=128, num_bands=16, ngram_size=2,
                                                similarity_threshold=similarity_threshold)
    distinct_indexes = select_distinct_indexes(duplicate_clusters, len(dedup_texts))
    return [data[idx] for idx in distinct_indexes]


def process_data(data: List[Dict], args: argparse.Namespace) -> List[Tuple[str, str, float]]:
    before_filter = len(data)
    data = deduplicate_data(data)
    print(f"Kept {len(data)} unique pairs out of {before_filter} total pairs")

    if not args.skip_language_check:
        before_filter = len(data)
        data = filter_english(data)
        print(f"Kept {len(data)} English pairs out of {before_filter} total pairs")
    else:
        print("Skipped language detection")


    if not args.skip_deduplication:
        before_filter = len(data)
        data = near_deduplicate_data(data, args.similarity_threshold)
        print(f"Kept {len(data)} pairs out of {before_filter} after deduplication")
    else:
        print("Skipped near deduplication")

    instructions = [item['instruction'] for item in data]
    answers = [item['output'] for item in data]

    return rank_and_filter(
        args.model_path,
        instructions,
        answers,
        topk=args.topk,
        threshold=args.threshold,
        ratio=args.ratio,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


def format_output(filtered_pairs: List[Tuple[str, str, float]]) -> List[Dict]:
    return [
        {
            "instruction": instruction,
            "output": answer,
            "score": float(score)
        }
        for instruction, answer, score in filtered_pairs
    ]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter and rank instruction-answer pairs using StyleRanker")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained StyleRanker model")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to JSON file containing instruction-answer pairs")
    parser.add_argument("--output_file", type=str, default="filtered_pairs.json", help="Path to save filtered pairs")
    parser.add_argument("--topk", type=int, help="Number of top pairs to keep")
    parser.add_argument("--threshold", type=float, help="Score threshold for filtering")
    parser.add_argument("--ratio", type=float, help="Ratio of top pairs to keep (0.0 to 1.0)")
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                        help="Similarity threshold for near-duplicates")
    parser.add_argument("--vicuna_output", type=str, help="Path to save Vicuna format output")
    parser.add_argument("--skip_language_check", action="store_true", help="Skip language detection step")
    parser.add_argument("--skip_deduplication", action="store_true", help="Skip deduplication step")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length for tokenization")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for scoring")


    args = parser.parse_args()

    if sum(arg is not None for arg in [args.topk, args.threshold, args.ratio]) != 1:
        parser.error("Exactly one of --topk, --threshold, or --ratio must be specified")

    return args


def main():
    args = parse_arguments()

    data = load_data(args.input_file)
    filtered_pairs = process_data(data, args)

    output_data = format_output(filtered_pairs)
    save_data(output_data, args.output_file)

    print(f"Filtered pairs saved to {args.output_file}")
    print(f"Final number of pairs: {len(filtered_pairs)}")

    if args.vicuna_output:
        vicuna_data = convert_to_vicuna_format(output_data)
        save_data(vicuna_data, args.vicuna_output)
        print(f"Vicuna format data saved to {args.vicuna_output}")


if __name__ == "__main__":
    main()