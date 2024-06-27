import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from style_ranker.ranker.model import StyleRanker


def load_ranker(model_path, device='cuda'):
    """
    Load the StyleRanker model and tokenizer.
    """
    model = StyleRanker.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def score_pairs(model, tokenizer, instructions, answers, max_length=512, batch_size=32, device='cuda'):
    """
    Score instruction-answer pairs using the StyleRanker model.
    """
    model.eval()
    scores = []

    for i in tqdm(range(0, len(instructions), batch_size)):
        batch_instructions = instructions[i:i + batch_size]
        batch_answers = answers[i:i + batch_size]

        instruction_inputs = tokenizer(batch_instructions, return_tensors='pt', padding=True, truncation=True,
                                       max_length=max_length).to(device)
        answer_inputs = tokenizer(batch_answers, return_tensors='pt', padding=True, truncation=True,
                                  max_length=max_length).to(device)

        with torch.no_grad():
            batch_scores = model(
                instruction_inputs.input_ids,
                instruction_inputs.attention_mask,
                answer_inputs.input_ids,
                answer_inputs.attention_mask
            )

        scores.extend(batch_scores.squeeze().tolist())

    return scores


def rank_pairs(instructions, answers, scores):
    """
    Rank instruction-answer pairs based on their scores.
    """
    ranked_pairs = sorted(zip(instructions, answers, scores), key=lambda x: x[2], reverse=True)
    return ranked_pairs


def filter_pairs(ranked_pairs, topk=None, threshold=None, ratio=None):
    """
    Filter ranked pairs based on top-k, threshold, or ratio.

    Args:
    ranked_pairs (list): List of tuples (instruction, answer, score) sorted by score in descending order.
    topk (int, optional): Number of top pairs to keep.
    threshold (float, optional): Score threshold for filtering.
    ratio (float, optional): Ratio of top pairs to keep (0.0 to 1.0).

    Returns:
    list: Filtered list of (instruction, answer, score) tuples.

    Note: Exactly one of topk, threshold, or ratio should be specified.
    """
    if sum(arg is not None for arg in [topk, threshold, ratio]) != 1:
        raise ValueError("Exactly one of topk, threshold, or ratio must be specified")

    if topk is not None:
        return ranked_pairs[:topk]
    elif threshold is not None:
        return [(inst, ans, score) for inst, ans, score in ranked_pairs if score >= threshold]
    else:  # ratio
        num_to_keep = int(len(ranked_pairs) * ratio)
        return ranked_pairs[:num_to_keep]



def rank_and_filter(model_path, instructions, answers, topk=None, threshold=None, ratio=None, max_length=512, batch_size=32, device='cuda'):
    """
    Perform the complete ranking and filtering pipeline.

    Args:
    model_path (str): Path to the trained StyleRanker model.
    instructions (list): List of instruction strings.
    answers (list): List of answer strings.
    topk (int, optional): Number of top pairs to keep.
    threshold (float, optional): Score threshold for filtering.
    ratio (float, optional): Ratio of top pairs to keep (0.0 to 1.0).
    max_length (int): Maximum length for tokenization.
    batch_size (int): Batch size for scoring.

    Returns:
    list: Filtered list of (instruction, answer, score) tuples.
    """
    model, tokenizer = load_ranker(model_path, device=device)
    scores = score_pairs(model, tokenizer, instructions, answers, max_length, batch_size, device=device)
    ranked_pairs = rank_pairs(instructions, answers, scores)
    filtered_pairs = filter_pairs(ranked_pairs, topk, threshold, ratio)
    return filtered_pairs

