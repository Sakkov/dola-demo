import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import re
from tqdm import tqdm

def _run_judge_inference(judge_model, judge_tokenizer, prompt, judge_device, verbose):
    """Helper function to run inference on the judge model."""
    judge_inputs = judge_tokenizer(prompt, return_tensors="pt").to(judge_device)
    judge_outputs = judge_model.generate(
        **judge_inputs,
        max_new_tokens=(50 if verbose < 2 else 3), 
        temperature=0.7, 
        top_p=0.8, 
        top_k=20, 
        pad_token_id=judge_tokenizer.eos_token_id,
        stop_strings=(["\n", "Statement", "Justification", "#"] if verbose < 2 else None), 
        tokenizer=judge_tokenizer,
    )
    judge_text = judge_tokenizer.batch_decode(judge_outputs[:, judge_inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0].strip()
    return judge_text

def _parse_comparison_output(judge_output):
    """Extracts and returns the score and any parsing error message from the judge's output."""
    match = re.search(r"([1-5])", judge_output)
    if match:
        score = int(match.group(1))
        return score, None
    else:
        score = None
        error_message = f"Warning: Could not parse score from judge output: '{judge_output}'"
        return score, error_message

def _parse_yes_no_output(judge_output):
    """Extracts and returns the score and any parsing error message from the judge's output."""
    match = re.search(r"(yes|no)", judge_output)
    if match:
        score = 1 if match.group(1) == "yes" else 0
        return score, None
    else:
        score = None
        error_message = f"Warning: Could not parse score from judge output: '{judge_output}'"
        return score, error_message
    
def comparison_judge(
        reference_statements,
        dola_answer,
        baseline_answer,
        judge_model,
        judge_tokenizer,
        judge_prompt_template,
        judge_device,
        verbose,
        i,
        display_example,
        scores_dola,
        scores_baseline,
        question,
    ):
    """Runs comparison-based evaluation for a single sample."""
    for statement in reference_statements:
            if verbose >= 2 and display_example:
                print(f"\n  --- Judging Sample {i+1}, Reference: '{statement}' ---")
                print(f"    Question: {question}")
                print(f"    DoLa Answer: {dola_answer}")
                print(f"    Baseline Answer: {baseline_answer}")
            
            judge_prompt_dola = judge_prompt_template.format(statement_1=statement, statement_2=dola_answer)
            judge_prompt_baseline = judge_prompt_template.format(statement_1=statement, statement_2=baseline_answer)
            
            if verbose >= 4:
                print(f"\n    --- Judge Prompts for Sample {i+1}, Ref Statement: '{statement}' ---")
                print(f"    Question: {question}")
                print(f"    Judge Prompt (DoLa):\n{judge_prompt_dola}\n")
                print(f"    Judge Prompt (Baseline):\n{judge_prompt_baseline}\n")

            judge_output_dola = _run_judge_inference(judge_model, judge_tokenizer, judge_prompt_dola, judge_device, verbose)
            judge_output_baseline = _run_judge_inference(judge_model, judge_tokenizer, judge_prompt_baseline, judge_device, verbose)
            
            if verbose >= 2 and display_example:
                print(f"    Judge Output (DoLa): \n{judge_output_dola}\n")
                print(f"    Judge Output (Baseline): \n{judge_output_baseline}\n")
            
            dola_score_val, dola_err = _parse_comparison_output(judge_output_dola)
            if dola_err and verbose >= 1: print(dola_err)
            if dola_score_val is not None: scores_dola.append(dola_score_val)

            baseline_score_val, baseline_err = _parse_comparison_output(judge_output_baseline)
            if baseline_err and verbose >= 1: print(baseline_err)
            if baseline_score_val is not None: scores_baseline.append(baseline_score_val)

def binary_judge(
        dola_answer,
        baseline_answer,
        judge_model,
        judge_tokenizer,
        judge_prompt_template,
        judge_device,
        verbose,
        i,
        display_example,
        scores_dola,
        scores_baseline,
        question,
    ):
    """Runs true-false evaluation for a single sample."""
    if verbose >= 2 and display_example:
        print(f"\n  --- Judging Sample {i+1} ---")
        print(f"    Question: {question}")
        print(f"    DoLa Answer: {dola_answer}")
        print(f"    Baseline Answer: {baseline_answer}")
    
    judge_prompt_dola = judge_prompt_template.format(question=question, answer=dola_answer)
    judge_prompt_baseline = judge_prompt_template.format(question=question, answer=baseline_answer)
    
    if verbose >= 3:
        print(f"\n    --- Judge Prompts for Sample {i+1} ---")
        print(f"    Question: {question}")
        print(f"    Judge Prompt (DoLa):\n{judge_prompt_dola}\n")
        print(f"    Judge Prompt (Baseline):\n{judge_prompt_baseline}\n")

    judge_output_dola = _run_judge_inference(judge_model, judge_tokenizer, judge_prompt_dola, judge_device, verbose)
    judge_output_baseline = _run_judge_inference(judge_model, judge_tokenizer, judge_prompt_baseline, judge_device, verbose)
    
    if verbose >= 2 and display_example:
        print(f"    Judge Output (DoLa): \n{judge_output_dola}\n")
        print(f"    Judge Output (Baseline): \n{judge_output_baseline}\n")
    
    dola_score_val, dola_err = _parse_yes_no_output(judge_output_dola)
    if dola_err and verbose >= 1: print(dola_err)
    if dola_score_val is not None: scores_dola.append(dola_score_val)

    baseline_score_val, baseline_err = _parse_yes_no_output(judge_output_baseline)
    if baseline_err and verbose >= 1: print(baseline_err)
    if baseline_score_val is not None: scores_baseline.append(baseline_score_val)

def evaluate_with_ai_judge(
    eval_method,
    evaluation_results,
    judge_model_name,
    judge_prompt_template,
    device,
    bnb_quantization,
    verbose,
    display_indices,
):
    if not judge_model_name:
        if verbose >=1:
            print("\nAI Judge not configured (judge_model_name is None). Skipping AI Judge evaluation.")
        return evaluation_results

    if verbose >= 0:
        print("\n===================================")
        print("     AI Judge Evaluation     ")
        print("===================================\n")

    judge_device = device 

    if verbose >= 1:
        print(f"AI Judge Evaluation method: {eval_method}")

    if verbose >= 2:
        print(f"Loading judge tokenizer: {judge_model_name}...")
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token

    if verbose >= 2:
        print(f"Loading judge model: {judge_model_name}")

    if device == "cuda" and bnb_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if verbose >= 2:
            print("  Applying 4-bit BNB quantization to Judge model as CUDA is available and BNB quantization is enabled.")
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    elif device != "cuda" and bnb_quantization: # BNB on CPU is not typical, load float32
        if verbose >= 2:
            print("  CUDA not available. Loading Judge model in float32 precision on CPU.")
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_name,
            torch_dtype=torch.float32,
            device_map=None 
        )
    elif device == "cuda" and not bnb_quantization:
        if verbose >= 2:
            print("  Loading Judge model in bfloat16 precision on CUDA.")
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else: # CPU and no BNB
        if verbose >= 2:
            print("  Loading Judge model in float32 precision on CPU.")
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
    
    judge_model.eval()
    if verbose >= 2: print(f"Judge model '{judge_model_name}' loaded.\n")

    if verbose >= 1:
        print(f"Evaluating {len(evaluation_results)} generated answer pairs with the AI Judge ('{judge_model_name}')...")

    for i, sample_results in enumerate(tqdm(evaluation_results, disable=verbose == 0, desc=f"Evaluating Samples with ({judge_model_name})")):
        display_example = i in display_indices
        question = sample_results["question"]
        reference_statements = sample_results["reference_answers"]
        dola_answer = sample_results["dola_answer"]
        baseline_answer = sample_results["baseline_answer"]

        scores_dola = []
        scores_baseline = []

        if eval_method == "comparison":
            comparison_judge(
                reference_statements,
                dola_answer,
                baseline_answer,
                judge_model,
                judge_tokenizer,
                judge_prompt_template,
                judge_device,
                verbose,
                i,
                display_example,
                scores_dola,
                scores_baseline,
                question,
            )
        elif eval_method == "true-false":
            binary_judge(
                dola_answer,
                baseline_answer,
                judge_model,
                judge_tokenizer,
                judge_prompt_template,
                judge_device,
                verbose,
                i,
                display_example,
                scores_dola,
                scores_baseline,
                question,
            )
        else:
            if verbose >= 1:
                print("AI Judge evaluation method not specified using 'true-false' evaluator.")
            binary_judge(
                dola_answer,
                baseline_answer,
                judge_model,
                judge_tokenizer,
                judge_prompt_template,
                judge_device,
                verbose,
                i,
                display_example,
                scores_dola,
                scores_baseline,
                question,
            )

        sample_results["dola_judge_score"] = int(np.max(scores_dola)) if scores_dola else None
        sample_results["baseline_judge_score"] = int(np.max(scores_baseline)) if scores_baseline else None

        if display_example and verbose >= 2:
             print(f"\n  --- Max Judge Scores for Sample {i+1} ---")
             print(f"    Question: {question}")
             print(f"    Reference Statements: {reference_statements}")
             print(f"    DoLa Answer: {dola_answer} -> Max Judge Score: {sample_results['dola_judge_score']}")
             print(f"    Baseline Answer: {baseline_answer} -> Max Judge Score: {sample_results['baseline_judge_score']}")
             print(f"  --------------------------------------------\n")
    if verbose >= 1: print("\n") 
    
    return evaluation_results