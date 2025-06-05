import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import evaluate
import numpy as np
from tqdm import tqdm
from evaluation_logic.utils import suppress_transformers_warnings, get_display_indices, get_date_and_index, load_model_and_tokenizer
from evaluation_logic.prompts import ANSWERING_PROMPT_TEMPLATE, JUDGE_PROMPT_TEMPLATE, JUDGE_PROMPT_TEMPLATE_TRUE_FALSE, JUDGE_PROMPT_TEMPLATE_TRUE_FALSE_SIMPLE
from evaluation_logic.ai_judge import evaluate_with_ai_judge

suppress_transformers_warnings()

def run_truthfulqa_evaluation(
    model_name="huggyllama/llama-7b",
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens=50,
    repetition_penalty=None,
    num_samples_to_test=817,
    num_examples_to_display=10,
    evaluation_metric_name="rouge",
    bnb_quantization=True,
    judge_model_name=None,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
    top_k=0,
    # dola_layers_setting removed
    prompt_template=ANSWERING_PROMPT_TEMPLATE,
    verbose=1,
    stop_strings=["Q:"],
    judge_prompt_template=JUDGE_PROMPT_TEMPLATE_TRUE_FALSE,
    judge_method="true-false",
):
    # Report the configuration parameters
    if verbose >= 0:
        print("\n===================================")
        print("        Configuration        ")
        print("===================================\n")
        print(f"  Device: {device}")
        print(f"  Number of Samples: {num_samples_to_test}")
        print(f"  Evaluation Metric: {evaluation_metric_name if not judge_model_name else 'AI Judge'}")
        if verbose >= 1:
            print(f"  Number of Examples to Display: {num_examples_to_display}")
        print(f"  Verbosity Level: {verbose}\n")
        print(f"  QA Model: {model_name}")
        print(f"  QA Max New Tokens: {max_new_tokens}")
        print(f"  QA Do Sample: {do_sample}")
        print(f"  QA Repetition Penalty: {repetition_penalty}")
        print(f"  QA BNB Quantization: {bnb_quantization}")
        # print(f"  QA DoLa Layers Setting: ...") # Removed
        if verbose >= 2:
            print(f"  QA Stop Strings: {stop_strings}")
            print(f"  QA Prompt template: \n{prompt_template}\n")
        print(f"  AI Judge Model: {judge_model_name if judge_model_name else 'Not using AI Judge'}")
        if judge_model_name and verbose >= 2:
            print(f"  Judge Prompt template: \n{judge_prompt_template}")
        print("\n===================================\n")

    # 1. Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device, bnb_quantization, verbose)

    # 2. Load TruthfulQA Dataset
    if verbose >= 2:
        print("Loading TruthfulQA dataset (generation task)...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")


    # 3. Load Evaluation Metric
    if judge_model_name:
        if verbose >= 2:
            print(f"\nUsing a Judge-AI ('{judge_model_name}') to evaluate semantic similarity.")
    else:
        if verbose >= 2:
            print(f"\nLoading {evaluation_metric_name} metric...")
        metric = evaluate.load(evaluation_metric_name)


    # 4. Iterate, Generate, and Evaluate
    if verbose >= 1:
        print(f"Generating answers for {num_samples_to_test} samples from TruthfulQA...")

    # Store results for later evaluation
    evaluation_results = []

    # Determine which examples to display
    # Ensure display_indices are within the bounds of num_samples_to_test
    display_indices = get_display_indices(num_samples_to_test, num_examples_to_display)
    if verbose >= 2 and num_examples_to_display > 0:
        print(f"Will display detailed examples for indices: {display_indices}\n")

    for i in tqdm(range(num_samples_to_test), disable=verbose == 0, desc=f"Generating Answers with {model_name}"):
        display_example = i in display_indices
        sample = dataset[i]
        question = sample["question"]
        reference_answers = sample["correct_answers"] # Best answer also available

        if display_example and verbose >= 2:
            print(f"\n-------------------- Sample {i+1}/{num_samples_to_test} --------------------")
            print(f"  Question: {question}")
            print(f"  Reference Answers:\n    {reference_answers}")

        prompt = prompt_template.format(question=question)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # --- Generation with DoLa --- # -> REMOVED

        # --- Baseline Generation (No DoLa) --- # -> This is now the only generation
        if display_example and verbose >= 2:
            print("Generating with Baseline (No DoLa)...")
        outputs_baseline = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=stop_strings,
            tokenizer=tokenizer,
        )
        answer_baseline = tokenizer.batch_decode(outputs_baseline[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0].strip().split('\n')[0]
        if display_example and verbose >= 2:
            print(f"  Baseline Answer: {answer_baseline}")
            print(f"--------------------------------------------------------\n")


        # Store results for this sample
        evaluation_results.append({
            "question": question,
            "reference_answers": reference_answers,
            # "dola_answer": answer_dola,
            "baseline_answer": answer_baseline,
            # "dola_judge_score": None,
            "baseline_judge_score": None,
        })

    # 4. AI Judge Evaluation
    evaluation_results = evaluate_with_ai_judge(
        eval_method=judge_method,
        evaluation_results=evaluation_results, # Will now only contain baseline_answer and expect baseline_judge_score
        judge_model_name=judge_model_name,
        judge_prompt_template=judge_prompt_template,
        device=device,
        bnb_quantization=False, # Causes problems with evaluation especially for fine-tuned mode
        verbose=verbose,
        display_indices=display_indices,
    )

    # 5. Report Aggregate Results
    if verbose >= 0:    
        print("\n===================================")
        print("      Evaluation Summary      ")
        print("===================================\n")


        if judge_model_name:
            # dola_scores = [r["dola_judge_score"] for r in evaluation_results if r["dola_judge_score"] is not None]
            baseline_scores = [r["baseline_judge_score"] for r in evaluation_results if r["baseline_judge_score"] is not None]
            print(f"\nAI Judge Results (using {judge_model_name}):")

            # if dola_scores:
            #     avg_dola_score = np.mean(dola_scores)
            #     print(f"  Average DoLa Judge-Score: {avg_dola_score:.2f}")
            # else:
            #     print("  No valid scores found for DoLa answers.")

            if baseline_scores:
                avg_baseline_score = np.mean(baseline_scores)
                print(f"  Average Baseline Judge-Score: {avg_baseline_score:.2f}")
            else:
                print("  No valid scores found for Baseline answers.")

            # if dola_scores and baseline_scores:
            #     print(f"  DoLa Improvement over Baseline: {(avg_dola_score - avg_baseline_score):.2f}")
            # else:
            #     print("  Unable to calculate improvement due to missing scores.")
            print() # Add a newline for spacing
        else:
            print("\nAI Judge was not configured (judge_model_name is None). No judge evaluation performed.")

        if verbose >= 1:
            print("\n===================================")
            print("         Test Complete         ")
            print("===================================\n")

    # Return the results
    return evaluation_results

def run_many(
    N = 3,
    model_name="huggyllama/llama-7b",
    max_new_tokens=50,
    repetition_penalty=None,
    num_samples_to_test=100, # Max 817 the size of the benchmark dataset
    num_examples_to_display=10,
    evaluation_metric_name="rouge",
    bnb_quantization=True,
    judge_model_name="Qwen/Qwen3-14B",
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
    top_k=0,
    # dola_layers_setting removed
    prompt_template=ANSWERING_PROMPT_TEMPLATE,
    verbose=1,
    stop_strings=["Q:"],
    judge_prompt_template=JUDGE_PROMPT_TEMPLATE_TRUE_FALSE,
    judge_method="true-false",
    
):
    """Run the same test multiple times and print out the aggregate results."""
    
    outputs = []
    evaluation_results = []
    for _ in range(N):
        evaluation_result = run_truthfulqa_evaluation(
            model_name=model_name,
            device=device, # device is not defined in this scope, should be passed or global
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            num_samples_to_test=num_samples_to_test,
            num_examples_to_display=num_examples_to_display,
            evaluation_metric_name=evaluation_metric_name,
            bnb_quantization=bnb_quantization,
            judge_model_name=judge_model_name,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # dola_layers_setting removed
            prompt_template=prompt_template,
            verbose=verbose,
            stop_strings=stop_strings,
            judge_prompt_template=judge_prompt_template,
            judge_method=judge_method,
        )
        outputs.append(output)
        evaluation_results.append(evaluation_result)
    
    # Print the averages
    # all_dola_scores = [] # Removed
    all_baseline_scores = []
    if evaluation_results and isinstance(evaluation_results[0], list): # Ensure it's a list of lists
        for single_run_result_list in evaluation_results:
            for sample_result in single_run_result_list:
                # if sample_result.get("dola_judge_score") is not None:
                #     all_dola_scores.append(sample_result["dola_judge_score"])
                if sample_result.get("baseline_judge_score") is not None:
                    all_baseline_scores.append(sample_result["baseline_judge_score"])

    if verbose >= 0:
        print("\n===================================")
        print("      Aggregate Results      ")
        print("===================================\n")
        

    aggregate_results = {
        # "dola_judge_score": None,
        "baseline_judge_score": None
    }
    # if all_dola_scores: # Removed
    #     aggregate_scores["dola_judge_score"] = np.mean(all_dola_scores)
    # else:
    #     if verbose >=0: print("  No valid scores found for DoLa answers across all runs.") # Adjusted print

    if all_baseline_scores:
        aggregate_results["baseline_judge_score"] = np.mean(all_baseline_scores)
    else:
        print("  No valid scores found for Baseline answers across all runs.")

    # if verbose >= 0:
        
    #     if all_dola_scores and all_baseline_scores:
    #         print(f"  DoLa Improvement over Baseline (across all runs): {(aggregate_results["dola_judge_score"] - aggregate_results["baseline_judge_score"]):.2f}")
    #     else:
    #         print("  Unable to calculate improvement due to missing scores across all runs.")
    #     print()

    

    return evaluation_results, aggregate_results

if __name__ == "__main__":

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ## Parameters
    verbose = -1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bnb_quantization = False
    runs_per_model = 1
    
    # Generation parameters
    models_to_test = [
        "huggyllama/llama-7b",
        # "mistralai/Mistral-7B-Instruct-v0.3",
        # "Qwen/Qwen3-1.7B",
        # "Qwen/Qwen3-14B",
    ]
    answering_prompt_template=ANSWERING_PROMPT_TEMPLATE
    max_new_tokens = 50
    repetition_penalty = 1.2
    num_samples_to_test = 817
    num_examples_to_display = 10
    do_sample=True
    temperature=0.9
    top_p=0.95
    top_k=0
    # dola_layers_settings removed
    stop_strings = ["Q:"]
    
    # AI Judge parameters
    judge_model_name = "allenai/truthfulqa-truth-judge-llama2-7B"
    judge_prompt_template = JUDGE_PROMPT_TEMPLATE_TRUE_FALSE_SIMPLE
    judge_method = "true-false"

    output = {
        "meta-config": {
            "models_to_test": models_to_test,
            "device": device,
            "seed": SEED,
            "answering_prompt_template": answering_prompt_template,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "num_samples_to_test": num_samples_to_test,
            "num_examples_to_display": num_examples_to_display,
            "bnb_quantization": bnb_quantization,
            "judge_model_name": judge_model_name,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            # "dola_layers_settings": dola_layers_settings, # Removed
            "verbose": verbose,
            "stop_strings": stop_strings,
            "judge_prompt_template": judge_prompt_template,
            "judge_method": judge_method,
            "runs_per_model": runs_per_model,
        },
        "results": [],
        "execution_time": 0,
        "packages": []
    }
    # list all packages and their versions
    import pkg_resources
    installed_packages = pkg_resources.working_set
    packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
    output["packages"] = packages_list

    import os
    import pathlib
    import time
    
    start_time = time.time()
    for i, model_name in enumerate(models_to_test):
        # Removed loop for dola_layers_settings
        
        # The 'run_many' function returns two main pieces of data:
        # 1. 'detailed_evals_N_runs': A list where each item is the detailed evaluation results from a single run
        #    by run_truthfulqa_evaluation (which itself is a list of dicts for each sample).
        #    e.g., [[{"question": ..., "baseline_answer": ..., ...}, ...]_run1, ..._runN]
        # 2. 'aggregate_scores_N_runs': A dictionary containing the aggregate scores (e.g., average judge scores) across all N runs.
        #    e.g., {"baseline_judge_score": avg_baseline_score_over_N_runs}
        
        if verbose >= 0:
            print(f"\n===== Starting Evaluation for Model: {model_name} =====")

        detailed_evals_N_runs, aggregate_scores_N_runs = run_many(
            N = runs_per_model,
            model_name=model_name,
            # device=device, # device is implicitly used by run_truthfulqa_evaluation from its own definition
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            num_samples_to_test=num_samples_to_test,
            num_examples_to_display=num_examples_to_display,
            evaluation_metric_name="rouge", # Still passed, though not used if judge is active
            bnb_quantization=bnb_quantization,
            judge_model_name=judge_model_name,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # dola_layers_setting removed
            prompt_template=answering_prompt_template,
            verbose=verbose,
            stop_strings=stop_strings,
            judge_prompt_template=judge_prompt_template,
            judge_method=judge_method,
        )
        output["results"].append({ # Changed structure here
            "model_name": model_name,
            "aggregate_scores_across_N_runs": aggregate_scores_N_runs,
            "detailed_evaluations_from_N_runs": detailed_evals_N_runs,
        })
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {int(total_time // 3600)}:{int((total_time % 3600) // 60)}:{int(total_time % 60)}\n")
    output["execution_time"] = total_time

    # Define and create the output directory using an absolute path
    script_dir = pathlib.Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    os.makedirs(output_dir, exist_ok=True)

    # Construct a more informative filename
    model_name_part = "NoModel"
    if models_to_test:
        if len(models_to_test) == 1:
            model_name_part = models_to_test[0].replace('/', '_')
        else:
            model_name_part = f"{len(models_to_test)}models"

    judge_name_part = "NoJudge"
    if judge_model_name:
        judge_name_part = judge_model_name.replace('/', '_')
    
    date_part, next_index = get_date_and_index(output_dir)

    # Assemble the filename parts
    filename_base = f"{model_name_part}_baseline_judge_{judge_name_part}_samples{num_samples_to_test}_runs{runs_per_model}_{date_part}_{next_index}"
    output_filename = f"{filename_base}.json"
    output_file_path = os.path.join(output_dir, output_filename)

    import json
    with open(output_file_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Results saved to {output_file_path}")
    
