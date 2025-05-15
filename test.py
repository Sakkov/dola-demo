from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from tqdm import tqdm
import numpy
import matplotlib.pyplot as plt

# CONFIGURATION
MODEL_NAME = "Qwen/Qwen3-4B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 20
REPETITION_PENALTY = 1.2

DOLA_LAYERS_SETTING = "high"


# Define the template
PROMPT_TEMPLATE = """Answer the following question in short. Do not give explanations only the answer.
Question: {question}
Answer:
"""

# 1. Initialize your model and tokenizer
model_name = MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Device set to use {device}")


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Set pad token if not present
    model.config.pad_token_id = model.config.eos_token_id

# 2. Prepare inputs
question = "Can people who have been consumed by a wolf be rescued?"
prompt = PROMPT_TEMPLATE.format(question=question)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 3. Call model.generate() directly with DoLa and dictionary return flags
# These parameters will be used to create a GenerationConfig internally
# or you can create and pass a GenerationConfig object explicitly.
generate_kwargs = {
    "dola_layers": DOLA_LAYERS_SETTING,
    "return_dict_in_generate": True,
    "output_attentions": True,
    "output_hidden_states": True,
    "output_scores": True,
    "output_logits": True,
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "repetition_penalty": REPETITION_PENALTY,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id
}


print("\nCalling model.generate() directly...")
outputs = model.generate(**inputs, **generate_kwargs)

# 4. Inspect the output

# Get generated token IDs and decode them
input_ids_length = inputs.input_ids.shape[1]
generated_sequence_ids = outputs.sequences[0, input_ids_length:]

if not generated_sequence_ids.tolist():
    print("No tokens were generated.")
    decoded_tokens = []
    scores_for_generated_tokens = []
    _scores = numpy.array([])
else:
    # Decode tokens for plot labels
    # Using tokenizer.decode for cleaner representation
    decoded_tokens = [tokenizer.decode([token_id]) for token_id in generated_sequence_ids.tolist()]

    # Extract the scores of the actual generated tokens
    # outputs.scores is a tuple of tensors
    print("Shape of the tuple of tensors")
    print(len(outputs.scores))
    print("Shape of one tensor")
    print(outputs.scores[0].shape)
    print(f"Vocabulary of the model {MODEL_NAME}")
    print(tokenizer.vocab_size)
    print("\nExtracting scores of generated tokens...")
    scores_for_generated_tokens = []
    for i, token_id in enumerate(generated_sequence_ids):
        # score_at_step_i is shape (1, vocab_size) as batch_size is 1 here
        score_at_step_i = outputs.scores[i]
        # Get the logit for the specific token_id that was generated
        token_logit = score_at_step_i[0, token_id.item()]
        scores_for_generated_tokens.append(token_logit.item())
    
    print(f"Scores of generated tokens: {scores_for_generated_tokens}")
    print()

    # Normalize the scores
    # Convert to numpy array for easier manipulation
    scores_np = numpy.array(scores_for_generated_tokens)
    min_score = numpy.min(scores_np)
    max_score = numpy.max(scores_np)

    if max_score == min_score:
        # Handle cases where all scores are the same (e.g., single token generated)
        # Set  scores to 0.5 to allow plotting a flat line
        _scores = numpy.full_like(scores_np, 0.5, dtype=float)
    else:
        _scores = scores_np
    
    print(f" scores: {_scores.tolist()}")

# Plot the  scores
if decoded_tokens and _scores.size > 0: # Check if there's anything to plot
    plt.figure(figsize=(12, 7)) # Adjusted figsize for potentially longer token labels
    plt.plot(range(len(_scores)), _scores, marker='o', linestyle='-', color='b')
    plt.xticks(range(len(decoded_tokens)), decoded_tokens, rotation=45, ha="right") # Use tokens as x-labels
    plt.title(' Scores of Generated Tokens')
    plt.xlabel('Generated Token')
    plt.ylabel(' Score of Generated Token')
    plt.grid(True)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig("scores_plot.png")
    print("\nPlot saved to scores_plot.png")
    plt.show()
else:
    print("\nSkipping plot as no tokens were generated or scores could not be processed.")
