import argparse
import json
from typing import List, Dict
from style_ranker.dedup import lsh_near_deduplication, select_distinct_indexes


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perform near-deduplication on documents")
    parser.add_argument("--input_file", type=str, required=True, help="Path to JSON file containing documents")
    parser.add_argument("--num_perm", type=int, default=128, help="Number of permutations for LSH")
    parser.add_argument("--num_bands", type=int, default=16, help="Number of bands for LSH")
    parser.add_argument("--ngram_size", type=int, default=2, help="N-gram size for tokenization")
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                        help="Similarity threshold for near-duplicates")
    parser.add_argument("--output_file", type=str, default="deduplicated_documents.json",
                        help="Path to save deduplicated documents")
    return parser.parse_args()


def load_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)


def save_data(data: List[Dict], file_path: str):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def create_documents(data: List[Dict]) -> List[str]:
    return [f"{item['instruction']} {item['output']}" for item in data]


def deduplicate(documents: List[str], args: argparse.Namespace) -> List[int]:
    duplicate_clusters = lsh_near_deduplication(
        documents,
        args.num_perm,
        args.num_bands,
        args.ngram_size,
        args.similarity_threshold
    )
    return select_distinct_indexes(duplicate_clusters, len(documents))


def main():
    args = parse_arguments()

    data = load_data(args.input_file)
    documents = create_documents(data)

    distinct_indexes = deduplicate(documents, args)
    output_data = [data[idx] for idx in distinct_indexes]

    save_data(output_data, args.output_file)

    print(f"Total documents: {len(documents)}")
    print(f"Deduplicated documents: {len(output_data)}")
    print(f"Deduplicated documents saved to {args.output_file}")


if __name__ == "__main__":
    main()