import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import re
from tqdm import tqdm

def _run_judge_inference(judge_model, judge_tokenizer, prompt, judge_device, verbose):
    """Helper function to run inference on the judge model."""
    judge_inputs = judge_tokenizer(prompt, return_tensors="pt").to(judge_device)
    judge_outputs_gen = judge_model.generate(
        **judge_inputs,
        max_new_tokens=(50 if verbose >= 2 else 3), 
        temperature=0, 
        top_p=0.8, 
        top_k=20, 
        pad_token_id=judge_tokenizer.eos_token_id,
        stop_strings=(["\n", "Statement", "Justification", "#"] if verbose < 2 else None), 
        tokenizer=judge_tokenizer,
        return_dict_in_generate=True,
        output_logits=True,
    )
    # Decode generated text
    generated_sequence = judge_outputs_gen.sequences[:, judge_inputs.input_ids.shape[-1]:]
    judge_text = judge_tokenizer.batch_decode(generated_sequence, skip_special_tokens=True)[0].strip()
    
    # Return a dictionary as expected by the calling functions
    return {
        "generated_text": judge_text,
        "logits": judge_outputs_gen.logits # Tuple of logit tensors for each generated token
    }    

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

def _parse_yes_no_output(first_token_logits: torch.Tensor, judge_tokenizer) -> tuple[int | None, str | None]:
    """
    Extracts and returns a binary score (1 for Yes/True, 0 for No/False) and 
    any parsing error message from the judge's first-token logits.

    Args:
        first_token_logits: A PyTorch tensor of shape (batch_size, vocab_size)
                            (typically (1, vocab_size)) containing the logits
                            for the first generated token.
        judge_tokenizer: The tokenizer used with the judge model.

    Returns:
        A tuple (score, error_message).
        - score: 1 if 'Yes'/'True' is more likely, 0 if 'No'/'False' is more likely.
                 None if the decision is ambiguous or an error occurred.
        - error_message: A string describing the error, or None on success.
    """
    
    # Define base strings for "Yes-like" and "No-like" responses.
    # We will check these and their variations (with leading space, different cases).
    base_yes_strings = ["Yes", "True"]
    base_no_strings = ["No", "False"]

    # Generate variations (e.g., "yes", " Yes", "YES")
    candidate_yes_strings = []
    for s_base in base_yes_strings:
        candidate_yes_strings.extend([
            s_base, s_base.lower(), s_base.upper(),
            " " + s_base, " " + s_base.lower(), " " + s_base.upper()
        ])
    candidate_yes_strings = sorted(list(set(candidate_yes_strings))) # Unique, sorted

    candidate_no_strings = []
    for s_base in base_no_strings:
        candidate_no_strings.extend([
            s_base, s_base.lower(), s_base.upper(),
            " " + s_base, " " + s_base.lower(), " " + s_base.upper()
        ])
    candidate_no_strings = sorted(list(set(candidate_no_strings))) # Unique, sorted

    # Get token IDs for these strings, only if they are single tokens.
    yes_token_ids = []
    for s in candidate_yes_strings:
        # `encode` returns a list of token IDs. We only care if the string is a single token.
        token_ids = judge_tokenizer.encode(s, add_special_tokens=False)
        if len(token_ids) == 1:
            yes_token_ids.append(token_ids[0])
    yes_token_ids = list(set(yes_token_ids)) # Unique IDs

    no_token_ids = []
    for s in candidate_no_strings:
        token_ids = judge_tokenizer.encode(s, add_special_tokens=False)
        if len(token_ids) == 1:
            no_token_ids.append(token_ids[0])
    no_token_ids = list(set(no_token_ids)) # Unique IDs

    # Squeeze the logits tensor from (batch_size, vocab_size) to (vocab_size)
    # Assuming batch_size is 1.
    # Ensure it's on CPU for processing.
    logits = first_token_logits.squeeze(0).to(device='cpu', dtype=torch.float32)

    max_yes_logit = -float('inf')
    found_yes_candidate = False
    for token_id in yes_token_ids:
        if 0 <= token_id < logits.shape[-1]: # Check if token_id is within vocab size
            max_yes_logit = max(max_yes_logit, logits[token_id].item())
            found_yes_candidate = True
    
    max_no_logit = -float('inf')
    found_no_candidate = False
    for token_id in no_token_ids:
        if 0 <= token_id < logits.shape[-1]: # Check if token_id is within vocab size
            max_no_logit = max(max_no_logit, logits[token_id].item())
            found_no_candidate = True

    if not found_yes_candidate and not found_no_candidate:
        # Neither yes-like nor no-like tokens from our lists were found as single tokens,
        # or their IDs were invalid.
        predicted_token_id = torch.argmax(logits).item()
        predicted_token_text = judge_tokenizer.decode([predicted_token_id], skip_special_tokens=False)
        error_message = (
            f"Warning: None of the predefined 'Yes'-like or 'No'-like strings could be "
            f"mapped to valid single tokens in the judge's vocabulary. "
            f"The most likely first token was '{predicted_token_text}' (ID: {predicted_token_id}). "
            f"Cannot determine Yes/No score from logits."
        )
        return None, error_message

    if not found_yes_candidate: # Only "No" candidates found
        if found_no_candidate: # This check implies max_no_logit is a valid number
            return 0, None # Score 0 for "No"
        else: # Should logically not be reached due to the check above
             return None, "Error: Inconsistent state. No 'Yes' candidates, and 'No' candidates also seem invalid."


    if not found_no_candidate: # Only "Yes" candidates found
        if found_yes_candidate: # This check implies max_yes_logit is a valid number
            return 1, None # Score 1 for "Yes"
        else: # Should logically not be reached
            return None, "Error: Inconsistent state. No 'No' candidates, and 'Yes' candidates also seem invalid."

    # If we reach here, both found_yes_candidate and found_no_candidate are True.
    if max_yes_logit > max_no_logit:
        return 1, None  # Score 1 for "Yes"
    elif max_no_logit > max_yes_logit:
        return 0, None  # Score 0 for "No"
    else:
        # Logits are equal, decision is ambiguous.
        predicted_token_id = torch.argmax(logits).item()
        predicted_token_text = judge_tokenizer.decode([predicted_token_id], skip_special_tokens=False)
        error_message = (
            f"Warning: Ambiguous Yes/No. The maximum logit for 'Yes'-like tokens ({max_yes_logit:.2f}) "
            f"is equal to the maximum logit for 'No'-like tokens ({max_no_logit:.2f}). "
            f"The overall most-likely first token was '{predicted_token_text}' (ID: {predicted_token_id})."
        )
        return None, error_message
    
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

            judge_output_dola_dict = _run_judge_inference(judge_model, judge_tokenizer, judge_prompt_dola, judge_device, verbose)
            judge_output_baseline_dict = _run_judge_inference(judge_model, judge_tokenizer, judge_prompt_baseline, judge_device, verbose)

            # Parse the dict
            judge_output_dola = judge_output_dola_dict["generated_text"]
            judge_output_baseline = judge_output_baseline_dict["generated_text"]
            
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

    judge_output_dola_dict = _run_judge_inference(judge_model, judge_tokenizer, judge_prompt_dola, judge_device, verbose)
    judge_output_baseline_dict = _run_judge_inference(judge_model, judge_tokenizer, judge_prompt_baseline, judge_device, verbose)

    judge_output_dola = judge_output_dola_dict["generated_text"]
    judge_output_baseline = judge_output_baseline_dict["generated_text"]

    judge_first_token_scores_dola = judge_output_dola_dict["logits"][0]
    judge_first_token_scores_baseline = judge_output_baseline_dict["logits"][0]

    if verbose >= 2 and display_example:
        print(f"    Judge Output (DoLa): \n{judge_output_dola}\n")
        print(f"    Judge Output (Baseline): \n{judge_output_baseline}\n")
    
    dola_score_val, dola_err = _parse_yes_no_output(judge_first_token_scores_dola, judge_tokenizer)
    if dola_err and verbose >= 1: print(dola_err)
    if dola_score_val is not None: scores_dola.append(dola_score_val)

    baseline_score_val, baseline_err = _parse_yes_no_output(judge_first_token_scores_baseline, judge_tokenizer)
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