import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import evaluate
import numpy as np
from tqdm import tqdm

# CONFIGURATION
MODEL_NAME = "Qwen/Qwen3-8B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 60
REPETITION_PENALTY = 1.2
NUM_SAMPLES_TO_TEST = 817
NUM_EXAMPLES_TO_DISPLAY = 10
EVALUATION_METRIC_NAME = "rouge"

DOLA_LAYERS_SETTING = "high"

# Define the template
PROMPT_TEMPLATE = """Answer the following question in short. Do not give explanations only the answer.
Question: {question}
Answer:
"""

def run_truthfulqa_evaluation():
    # Report the configuration parameters
    print("\n--- Configuration ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Max New Tokens: {MAX_NEW_TOKENS}")
    print(f"Repetition Penalty: {REPETITION_PENALTY}")
    print(f"Number of Samples: {NUM_SAMPLES_TO_TEST}")
    print(f"Evaluation Metric: {EVALUATION_METRIC_NAME}")
    print(f"DoLa Layers Setting: {DOLA_LAYERS_SETTING}")
    print(f"Prompt template: \n{PROMPT_TEMPLATE}")
    print("---\n")

    # 1. Load Model and Tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {MODEL_NAME}")
    if DEVICE == "cuda":
        # Configure BitsAndBytesConfig for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Applying 4-bit BNB quantization as CUDA is available.")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        print("CUDA not available. Loading model in default precision (float32 for CPU).")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
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
    print(f"Loading {EVALUATION_METRIC_NAME} metric...")
    metric = evaluate.load(EVALUATION_METRIC_NAME)


    # 4. Iterate, Generate, and Evaluate
    print(f"Testing on {NUM_SAMPLES_TO_TEST} samples from TruthfulQA...")

    dola_scores = []
    baseline_scores = []
    display_indices = np.linspace(0, NUM_SAMPLES_TO_TEST - 1, NUM_EXAMPLES_TO_DISPLAY, dtype=int)
    print(f"Displaying examples at indices: {display_indices}")

    for i in tqdm(range(NUM_SAMPLES_TO_TEST)):
        display_example = i in display_indices
        sample = dataset[i]
        question = sample["question"]
        reference_answers = sample["correct_answers"] # Best answer also available

        if display_example:
            print(f"\n--- Sample {i+1}/{NUM_SAMPLES_TO_TEST} ---")
            print(f"Question: {question}")
            print(f"Reference Answers:\n{reference_answers}")

        prompt = PROMPT_TEMPLATE.format(question=question)

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # --- Generation with DoLa ---
        if display_example:
            print(f"Generating with DoLa (dola_layers='{DOLA_LAYERS_SETTING}')...")
        outputs_dola = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            dola_layers=DOLA_LAYERS_SETTING,
            do_sample=False, # Deterministic output for comparison
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=REPETITION_PENALTY,
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
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=REPETITION_PENALTY,
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