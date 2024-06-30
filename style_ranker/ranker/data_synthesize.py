import argparse
import json
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from typing import List, Dict, Any

from style_ranker.consts import STYLE_PROMPTS


def create_prompt(instruction: str, reference_answer: str, prompt_template: str) -> str:
    if prompt_template not in STYLE_PROMPTS:
        raise ValueError(f"Invalid prompt template: {prompt_template}")

    if prompt_template == 'referenced':
        return STYLE_PROMPTS[prompt_template].format(
            instruction=instruction,
            reference_answer=reference_answer
        )
    elif prompt_template == 'direct':
        return STYLE_PROMPTS[prompt_template].format(
            instruction=instruction
        )
    else:
        raise ValueError(f"Invalid prompt template: {prompt_template}")


def batch_data(data: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def get_chatgpt_response(input_tuple):
    client, model, prompt, item = input_tuple
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip(), item
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return None, item


def process_batch(batch: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    client = OpenAI(api_key=args.api_key)
    results = []

    prompts = [
        create_prompt(
            item['instruction'],
            item['output'],
            args.prompt_template
        ) for item in batch
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_chatgpt_response, (client, args.model, prompt, item))
                   for prompt, item in zip(prompts, batch)]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                output, item = result
                # Preserve the original item structure and update only the 'output' field
                updated_item = item.copy()
                updated_item['output'] = output
                results.append(updated_item)
            else:
                # If API call failed, keep the original item unchanged
                results.append(item)
    # Sort the results based on the order of instructions in the original batch
    batch_instructions = [item['instruction'] for item in batch]
    sorted_results = sorted(results, key=lambda x: batch_instructions.index(x['instruction']))
    return sorted_results


def process_data(args: argparse.Namespace):
    with open(args.input_file, 'r') as f:
        code_json = json.load(f)

    new_data = []

    for batch in tqdm(batch_data(code_json, args.batch_size),
                      total=(len(code_json) + args.batch_size - 1) // args.batch_size):
        try:
            new_data.extend(process_batch(batch, args))
        except Exception as e:
            print(f'Error processing batch: {str(e)}')
            # If a batch fails, add the original items to maintain data integrity
            new_data.extend(batch)

    # Final check to ensure overall order is preserved
    assert len(new_data) == len(code_json), "Output length doesn't match input length"
    for i, (new_item, original_item) in enumerate(zip(new_data, code_json)):
        assert new_item['instruction'] == original_item['instruction'], f"Order mismatch at index {i}"

    with open(args.output_file, 'w') as f:
        json.dump(new_data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Process code data with ChatGPT for style consistency")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--prompt_template", type=str, required=True, choices=['referenced', 'direct'],
                        help="Prompt template to use")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model to use for ChatGPT")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing")

    args = parser.parse_args()
    process_data(args)


if __name__ == "__main__":
    main()