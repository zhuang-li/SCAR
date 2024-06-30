TASK_PROMPTS = {
    "helpfulness": """
You are an expert evaluator assessing the helpfulness of instruction-following responses. Your task is to rate the helpfulness on a scale of 0-9.

Instruction: {PROBLEM}

Response: {OUTPUT}

To evaluate the helpfulness, follow these steps:
1. Understand the instruction thoroughly.
2. Analyze how well the response addresses the instruction.
3. Consider the relevance, depth, and clarity of the information provided.
4. Assess whether the response provides actionable or insightful information.
5. Determine if there are any missing crucial elements.

Helpfulness Rating Scale (0-9):
0-1: Completely irrelevant or no meaningful response
2-3: Barely addresses the instruction, missing crucial elements
4-5: Partially addresses the instruction, but lacks depth or clarity
6-7: Adequately addresses the instruction with useful information
8-9: Fully addresses the instruction, providing comprehensive and insightful information

Now, think through your evaluation step-by-step:

1. Instruction understanding:
[Your thoughts here]

2. Response analysis:
[Your thoughts here]

3. Relevance, depth, and clarity:
[Your thoughts here]

4. Actionable or insightful information:
[Your thoughts here]

5. Missing elements:
[Your thoughts here]

Based on your step-by-step analysis, provide your final evaluation:

Final Helpfulness Score (0-9): Please provide a single numeric score between 0 and 9.
""",

    "correctness": """
You are an expert evaluator assessing the correctness of instruction-following responses. Your task is to rate the correctness on a scale of 0-9.

Instruction: {PROBLEM}

Response: {OUTPUT}

To evaluate the correctness, follow these steps:
1. Understand the instruction and the expected correct response.
2. Analyze the factual accuracy of the response.
3. Identify any errors, including factual inaccuracies, logical flaws, or execution errors if the response involves code.
4. Assess the completeness of the response.
5. Consider the precision and specificity of the information provided.

Correctness Rating Scale (0-9):
0-1: Entirely incorrect or nonsensical, major factual errors or execution failures
2-3: Mostly incorrect with significant factual inaccuracies or major execution errors
4-5: Partially correct, but contains significant mistakes, omissions, or execution errors
6-7: Mostly correct with minor inaccuracies or execution errors
8-9: Completely correct and accurate information, no execution errors

Now, think through your evaluation step-by-step:

1. Expected correct response:
[Your thoughts here]

2. Factual accuracy and execution analysis:
[Your thoughts here]

3. Errors or misconceptions:
[Your thoughts here]

4. Completeness assessment:
[Your thoughts here]

5. Precision and specificity:
[Your thoughts here]

Based on your step-by-step analysis, provide your final evaluation:

Final Correctness Score (0-9): Please provide a single numeric score between 0 and 9.
"""
}

SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {PROBLEM} ASSISTANT: {OUTPUT}{EOS_TOKEN}"

STYLE_PROMPTS = {
    'referenced': """
### Reference Answer:

{reference_answer}

### Background:

You are a knowledgeable AI assistant. Above is a reference answer. Below is an instruction that describes a task. Given the reference answer, write a response that appropriately completes the request. Please keep the semantics of the reference answer unchanged in your response, while crafting your final response as if you have never seen the reference answer. The phrase "reference answer" should never be mentioned in your response."

### Instruction:

{instruction}

### Response:
""",
    'direct': """
### Background:

You are a knowledgeable AI assistant. Below is an instruction that describes a task. Please write a response that appropriately completes the request.

### Instruction:

{instruction}

### Response:
"""
}