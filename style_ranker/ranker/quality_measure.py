import argparse
import concurrent.futures
import json
import random
import os
import re
from collections import Counter
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Optional
from openai import OpenAI

from style_ranker.consts import TASK_PROMPTS


def generate_prompt(aspect: str, problem: str, output: str) -> str:
    prompt = TASK_PROMPTS[aspect]
    return prompt.format(PROBLEM=problem, OUTPUT=output)


def process_score(content: str, aspect: str) -> Optional[float]:
    """
    Processes the content to extract the score for a given aspect.

    Args:
        content (str): The raw content from GPT response.
        aspect (str): The evaluation aspect (e.g., "helpfulness", "correctness").
    Returns:
        float: The extracted score as a float between 0 and 9, or None if no valid score is found.
    """
    # Normalize content: lowercase and remove parentheses
    content = content.lower().replace("(", "").replace(")", "").replace("-", "").replace("out of", "/")

    # Split content into lines
    splits = content.split("\n")

    # Extract lines containing "score" or the aspect keyword
    ls = [
        ll.strip(".").replace("out of ", "/").replace("/4", "")
        for l in splits
        for ll in l.lstrip('0123456789. ').split(". ")
        if any(item in ll for item in ["score", "final score"] + aspect.split())
    ]

    # Extract numeric values from the lines
    ans = [ll for l in ls for ll in l.split() if ll.isnumeric()]

    # Define additional patterns to match scores
    patterns = [
        r"final score:\s*(\d+(?:\.\d+)?)",  # Matches "Final Score: X"
        r"final\s*(\w*)\s*score\s*is\s*(\d+(?:\.\d+)?)",  # Matches "Final correctness score is X"
        r"(\d+(?:\.\d+)?)\s*\/\s*9",  # Matches "X/9"
        r"(\d+(?:\.\d+)?)\s*out\s*of\s*9",  # Matches "X out of 9"
        r"score\s*is\s*(\d+(?:\.\d+)?)",  # Matches "score is X"
        r"rating\s*is\s*(\d+(?:\.\d+)?)",  # Matches "rating is X"
        r"(\d+(?:\.\d+)?)\s*points?"  # Matches "X points"
    ]

    # Try to find scores using patterns
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            # If the pattern captures multiple groups, pick the group that is the score
            if isinstance(match, tuple):
                ans.extend([m for m in match if re.match(r'^\d+(?:\.\d+)?$', m)])
            else:
                ans.append(match)

    # Convert all found scores to floats and filter for valid range (0-9)
    valid_scores = [float(score) for score in ans if 0 <= float(score) <= 9]

    # If there are multiple valid scores, return the most common one
    if len(set(valid_scores)) != 1 and len(valid_scores) > 1:
        return float(Counter(valid_scores).most_common(1)[0][0])

    # Handle special cases where there are no valid scores
    if not valid_scores:
        if "n/a" in content:
            return 0.0
        return None

    # Return the single valid score
    return valid_scores[0] if valid_scores else None


def evaluate(problem: str, output: str, aspect: str, model: str) -> Tuple[Optional[float], str]:
    prompt = generate_prompt(aspect, problem, output)
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024
        )
        content = response.choices[0].message.content
        score = process_score(content, aspect)
        return score, content
    except Exception as e:
        return None, f"Error: {str(e)}"


def get_chatgpt_response(input_tuple: Tuple[str, str, str, str]) -> Dict[str, Any]:
    instruction, output, aspect, model = input_tuple
    score, content = evaluate(problem=instruction, output=output, aspect=aspect, model=model)
    return {
        'instruction': instruction,
        'output': output,
        'score': score,
        'content': content
    }


def process_file(file_path: str, aspect: str, model: str, sample_size: Optional[int] = None, batch_size: int = 10) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    with open(file_path, 'r') as f:
        data = json.load(f)

    if sample_size:
        random.seed(42)
        data = random.sample(data, min(sample_size, len(data)))

    inputs = [(item['instruction'], item['output'], aspect, model) for item in data]

    data_with_scores = []
    score_dict = {}

    # Process in batches
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:i + batch_size]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for result in executor.map(get_chatgpt_response, batch):
                data_with_scores.append(result)
                key = f"Instruction: {result['instruction']}Answer: {result['output']}"
                # set the score to 0 if it is None
                score_dict[key] = result['score'] if result['score'] is not None else 0.0


    return data_with_scores, score_dict

def main(args):
    os.environ["OPENAI_API_KEY"] = args.api_key

    for file_name in args.file_list:
        for aspect in args.aspects:
            print(f"Processing {file_name} for {aspect}...")
            file_path = os.path.join(args.data_dir, file_name)
            mode = os.path.splitext(file_name)[0]

            try:
                data_with_scores, score_dict = process_file(file_path, aspect, args.model, args.sample_size, args.batch_size)

                # Save the first type of output file with the specified naming convention
                aspect_short = "help" if aspect == "helpfulness" else "correct"
                output_file = os.path.join(args.output_dir, f'{mode}_{aspect_short}_score_raw.json')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(data_with_scores, f, indent=4)
                print(f"Results saved to {output_file}")

                # Save the second type of output file with the specified naming convention
                output_file_dict = os.path.join(args.output_dir, f'{mode}_{aspect_short}_score.json')
                with open(output_file_dict, 'w') as f:
                    json.dump(score_dict, f, indent=4)
                print(f"Dictionary results saved to {output_file_dict}")

            except Exception as e:
                print(f"Error processing {file_name} for {aspect}: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate instruction-following responses using GPT models.")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing input files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--file_list", nargs='+', required=True, help="List of input files to process")
    parser.add_argument("--aspects", nargs='+', default=['helpfulness', 'correctness'], help="Aspects to evaluate")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="GPT model to use for evaluation")
    parser.add_argument("--sample_size", type=int, help="Optional: Number of samples to evaluate from each file")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing requests")

    args = parser.parse_args()
    main(args)