import os
import json
import torch
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from evaluation_logic.ai_judge import _run_judge_inference, _parse_yes_no_output
from evaluation_logic.prompts import JUDGE_PROMPT_TEMPLATE_TRUE_FALSE_SIMPLE, INFO_PROMPT_TEMPLATE_TRUE_FALSE_SIMPLE

QA_INPUT_FILE = "tfqa_result_no-dola-full.json"

JUDGE_MODEL_NAME = "allenai/truthfulqa-truth-judge-llama2-7B"
INFO_MODEL_NAME = "allenai/truthfulqa-info-judge-llama2-7B"

results = json.load(open(QA_INPUT_FILE, "r"))
questions = results["question"]
answers = results["model_completion"]

judge_model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)

scores = []
evaluation_results = []

judge_prompt_template = JUDGE_PROMPT_TEMPLATE_TRUE_FALSE_SIMPLE
judge_device = "cuda"

for question, answer in tqdm(zip(questions, answers)):
    judge_prompt = judge_prompt_template.format(question=question, answer=answer)

    judge_output_dict = _run_judge_inference(judge_model, judge_tokenizer, judge_prompt, judge_device, verbose=2)

    judge_output = judge_output_dict["generated_text"]

    judge_first_token_scores = judge_output_dict["logits"][0]
    
    score_val, err = _parse_yes_no_output(judge_first_token_scores, judge_tokenizer)
    if score_val is not None: scores.append(score_val)

    evaluation_results.append({
        "question": question,
        "answer": answer,
        "judge_output": judge_output,
        "judge_score": score_val
    })

score = statistics.fmean(scores)

# info_model = AutoModelForCausalLM.from_pretrained(INFO_MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
# info_tokenizer = AutoTokenizer.from_pretrained(INFO_MODEL_NAME)

# info_prompt_template = INFO_PROMPT_TEMPLATE_TRUE_FALSE_SIMPLE
# info_device = "cuda"

# for question, answer in tqdm(zip(questions, answers)):
#     info_prompt = info_prompt_template.format(question=question, answer=answer)

#     info_output_dict = _run_judge_inference(info_model, info_tokenizer, info_prompt, info_device, verbose=2)

#     info_output = info_output_dict["generated_text"]

#     info_first_token_scores = info_output_dict["logits"][0]

results = {
    "final score": score,
    "evaluation results": evaluation_results
}

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)    
