import json
import torch

def convert_to_vicuna_format(data):
    vicuna_data = []
    for idx, item in enumerate(data):
        vicuna_item = {
            "id": f"identity_{idx}",
            "conversations": [
                {
                    "from": "human",
                    "value": item['instruction']
                },
                {
                    "from": "gpt",
                    "value": item['output']
                }
            ]
        }
        vicuna_data.append(vicuna_item)
    return vicuna_data

def max_pooling(hidden_states, attention_mask):
    attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked_hidden_states = hidden_states.masked_fill(attention_mask == 0, -1e9)
    max_hidden_states, _ = torch.max(masked_hidden_states, dim=1)
    return max_hidden_states

def mean_pooling(hidden_states, attention_mask):
    attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked_hidden_states = hidden_states * attention_mask
    sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
    num_tokens = torch.sum(attention_mask, dim=1)
    mean_hidden_states = sum_hidden_states / num_tokens
    return mean_hidden_states

def filter_ins_ans_pairs(instructions, answer_list1, answer_list2, ins_ans1_score_dict, ins_ans2_score_dict, constraint_mode, min_thres, max_thres):
    cnt = 0
    index_list = []
    for ins, ans1, ans2 in zip(instructions, answer_list1, answer_list2):

        if constraint_mode == 'abs':

            if min(ins_ans1_score_dict["Instruction: " + ins + "Answer: " + ans1],
                   ins_ans2_score_dict["Instruction: " + ins + "Answer: " + ans2]) > min_thres and max(ins_ans1_score_dict["Instruction: " + ins + "Answer: " + ans1],
                   ins_ans2_score_dict["Instruction: " + ins + "Answer: " + ans2]) < max_thres:
                index_list.append(cnt)
        elif constraint_mode == 'abs_pos':
            if ins_ans1_score_dict["Instruction: " + ins + "Answer: " + ans1] > min_thres and ins_ans1_score_dict["Instruction: " + ins + "Answer: " + ans1] < max_thres:
                index_list.append(cnt)
        elif constraint_mode == 'diff':
            if abs(ins_ans1_score_dict["Instruction: " + ins + "Answer: " + ans1] - ins_ans2_score_dict[
                "Instruction: " + ins + "Answer: " + ans2]) < max_thres:
                index_list.append(cnt)
        elif constraint_mode == 'abs_diff':
            if abs(ins_ans1_score_dict["Instruction: " + ins + "Answer: " + ans1] - ins_ans2_score_dict[
                "Instruction: " + ins + "Answer: " + ans2]) < max_thres and \
                    min(ins_ans1_score_dict["Instruction: " + ins + "Answer: " + ans1], ins_ans2_score_dict[
                        "Instruction: " + ins + "Answer: " + ans2]) > min_thres:
                index_list.append(cnt)
        elif constraint_mode == 'none':
            index_list.append(cnt)
        cnt += 1
    return index_list



def load_and_process_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    instructions = [item['instruction'] for item in data]
    outputs = [item['output'] for item in data]
    return instructions, outputs



