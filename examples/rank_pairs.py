import argparse
import json
from typing import List, Dict, Tuple
from style_ranker.rank import rank_and_filter


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank instruction-answer pairs using StyleRanker")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained StyleRanker model")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to JSON file containing instruction-answer pairs")
    parser.add_argument("--output_file", type=str, default="ranked_pairs.json", help="Path to save ranked pairs")
    parser.add_argument("--topk", type=int, help="Number of top pairs to keep")
    parser.add_argument("--threshold", type=float, help="Score threshold for filtering")
    parser.add_argument("--ratio", type=float, help="Ratio of top pairs to keep (0.0 to 1.0)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length for tokenization")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for scoring")

    args = parser.parse_args()

    if sum(arg is not None for arg in [args.topk, args.threshold, args.ratio]) != 1:
        parser.error("Exactly one of --topk, --threshold, or --ratio must be specified")

    return args


def load_data(file_path: str) -> Tuple[List[str], List[str]]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    instructions = [item['instruction'] for item in data]
    answers = [item['output'] for item in data]
    return instructions, answers


def save_ranked_pairs(filtered_pairs: List[Tuple[str, str, float]], output_file: str):
    output_data = [
        {
            "rank": rank,
            "instruction": instruction,
            "output": answer,
            "score": float(score)
        }
        for rank, (instruction, answer, score) in enumerate(filtered_pairs, 1)
    ]
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Ranked pairs saved to {output_file}")


def main():
    args = parse_arguments()

    instructions, answers = load_data(args.input_file)

    filtered_pairs = rank_and_filter(
        args.model_path,
        instructions,
        answers,
        topk=args.topk,
        threshold=args.threshold,
        ratio=args.ratio,
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    save_ranked_pairs(filtered_pairs, args.output_file)


if __name__ == "__main__":
    main()