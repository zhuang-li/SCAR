import json

def load_and_process_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    instructions = [item['instruction'] for item in data]
    outputs = [item['output'] for item in data]
    return instructions, outputs
