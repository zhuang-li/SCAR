import json
import argparse

from style_ranker.utils import convert_to_vicuna_format


def convert_file_to_vicuna_format(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    vicuna_data = convert_to_vicuna_format(data)

    with open(output_file, 'w') as f:
        json.dump(vicuna_data, f, indent=2)

    print(f"Converted {len(vicuna_data)} items to Vicuna format and saved to {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert data to Vicuna format")
    parser.add_argument("--input_file", type=str, help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, help="Path to save Vicuna format output")
    args = parser.parse_args()

    convert_file_to_vicuna_format(args.input_file, args.output_file)