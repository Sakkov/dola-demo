import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import evaluate
import numpy as np
from tqdm import tqdm

def run_truthfulqa_evaluation(
    model_name="huggyllama/llama-7b",
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens=50,
    repetition_penalty=None,
    num_samples_to_test=817,
    num_examples_to_display=10,
    evaluation_metric_name="rouge",
    bnb_quantization=True,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
    top_k=0,
    dola_layers_setting=[0,2,4,6,8,10,12,14,32],
    prompt_template="""Answer the following question in short. Do not give explanations only the answer.
Question: {question}
Answer:
"""
):
    # Report the configuration parameters
    print("\n--- Configuration ---")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Repetition Penalty: {repetition_penalty}")
    print(f"Number of Samples: {num_samples_to_test}")
    print(f"Evaluation Metric: {evaluation_metric_name}")
    print(f"BNB Quantization: {bnb_quantization}")
    print(f"DoLa Layers Setting: {dola_layers_setting}")
    print(f"Prompt template: \n{prompt_template}")
    print("---\n")

    # 1. Load Model and Tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    if device == "cuda" and bnb_quantization:
        # Configure BitsAndBytesConfig for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Applying 4-bit BNB quantization as CUDA is available and BNB quantization is enabled.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    elif device != "cuda" and bnb_quantization:
        print("CUDA not available. Loading model in float32 precision on CPU.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
    elif device == "cuda" and not bnb_quantization:
        print("Loading model in bfloat16 precision on CUDA.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        print("Loading model in float32 precision on CPU.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
    
    # To check number of layers for DoLa:
    config = model.config
    print(f"Model has {config.num_hidden_layers} layers.")

    model.eval()

    # 2. Load TruthfulQA Dataset
    print("Loading TruthfulQA dataset (generation task)")
    dataset = load_dataset("truthful_qa", "generation", split="validation")


    # 3. Load Evaluation Metric
    print(f"Loading {evaluation_metric_name} metric...")
    metric = evaluate.load(evaluation_metric_name)


    # 4. Iterate, Generate, and Evaluate
    print(f"Testing on {num_samples_to_test} samples from TruthfulQA...")

    dola_scores = []
    baseline_scores = []
    display_indices = np.linspace(0, num_samples_to_test - 1, num_examples_to_display, dtype=int)
    print(f"Displaying examples at indices: {display_indices}")

    for i in tqdm(range(num_samples_to_test)):
        display_example = i in display_indices
        sample = dataset[i]
        question = sample["question"]
        reference_answers = sample["correct_answers"] # Best answer also available

        if display_example:
            print(f"\n--- Sample {i+1}/{num_samples_to_test} ---")
            print(f"Question: {question}")
            print(f"Reference Answers:\n{reference_answers}")

        prompt = prompt_template.format(question=question)

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # --- Generation with DoLa ---
        if display_example:
            print(f"Generating with DoLa (dola_layers='{DOLA_LAYERS_SETTING}')...")
        outputs_dola = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            dola_layers=dola_layers_setting,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )
        answer_dola = tokenizer.batch_decode(outputs_dola[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0].strip().split('\n')[0]
        if display_example:
            print(f"DoLa Answer: {answer_dola}")

        # Evaluate DoLa
        results = []
        results_sums = []
        for reference_answer in reference_answers:
            result = metric.compute(predictions=[answer_dola], references=[reference_answer])
            results.append(result)
            result_sum = np.sum(list(result.values()))
            results_sums.append(result_sum)
        best_result_index = np.argmax(results_sums)
        best_reference_answer = reference_answers[best_result_index]
        if display_example:
            print(f"Best Reference Answer: {best_reference_answer}")
        dola_results = np.max(results[best_result_index])
        dola_scores.append(dola_results)
        if display_example:
            print(f"DoLa ROUGE: {dola_results}\n")

        # --- Baseline Generation (No DoLa) ---
        if display_example:
            print("Generating with Baseline (No DoLa)...")
        outputs_baseline = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )
        answer_baseline = tokenizer.batch_decode(outputs_baseline[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0].strip().split('\n')[0]
        if display_example:
            print(f"Baseline Answer: {answer_baseline}")

        # Evaluate Baseline
        results = []
        results_sums = []
        for reference_answer in reference_answers:
            result = metric.compute(predictions=[answer_baseline], references=[reference_answer])
            results.append(result)
            result_sum = np.sum(list(result.values()))
            results_sums.append(result_sum)
        best_result_index = np.argmax(results_sums)
        best_reference_answer = reference_answers[best_result_index]
        if display_example:
            print(f"Best Reference Answer: {best_reference_answer}")
        baseline_results = np.max(results[best_result_index])
        baseline_scores.append(baseline_results)
        if display_example:
            print(f"Baseline ROUGE: {baseline_results}\n")

    # 5. Report Aggregate Results
    print("\n--- Evaluation Summary ---")

    if dola_scores:
        avg_dola_rouge_l = np.mean([score['rougeL'] for score in dola_scores if score])
        avg_dola_rouge_1 = np.mean([score['rouge1'] for score in dola_scores if score])
        avg_dola_rouge_2 = np.mean([score['rouge2'] for score in dola_scores if score])
        print("\nAverage DoLa Scores:")
        print(f"  ROUGE-L: {avg_dola_rouge_l:.4f}")
        print(f"  ROUGE-1: {avg_dola_rouge_1:.4f}")
        print(f"  ROUGE-2: {avg_dola_rouge_2:.4f}")
    else:
        print("\nNo DoLa scores were recorded (perhaps due to errors or no samples processed).")

    if baseline_scores:
        avg_baseline_rouge_l = np.mean([score['rougeL'] for score in baseline_scores if score])
        avg_baseline_rouge_1 = np.mean([score['rouge1'] for score in baseline_scores if score])
        avg_baseline_rouge_2 = np.mean([score['rouge2'] for score in baseline_scores if score])
        print("\nAverage Baseline Scores:")
        print(f"  ROUGE-L: {avg_baseline_rouge_l:.4f}")
        print(f"  ROUGE-1: {avg_baseline_rouge_1:.4f}")
        print(f"  ROUGE-2: {avg_baseline_rouge_2:.4f}")
    else:
        print("\nNo Baseline scores were recorded.")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    run_truthfulqa_evaluation()