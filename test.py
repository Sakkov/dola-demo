from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from tqdm import tqdm
import numpy
import matplotlib.pyplot as plt

# CONFIGURATION
MODEL_NAME = "Qwen/Qwen3-4B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 42
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

decoded_tokens = [] # Initialize for broader scope

if not generated_sequence_ids.tolist():
    print("No tokens were generated.")
    scores_for_generated_tokens = []
    _scores = numpy.array([])
else:
    # Decode tokens for plot labels
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

    # Convert to numpy array for plotting
    _scores = numpy.array(scores_for_generated_tokens)
    print(f"Final scores for plotting: {_scores.tolist()}")


# Plot the scores
if decoded_tokens and _scores.size > 0: # Check if there's anything to plot
    plt.figure(figsize=(12, 7)) 
    plt.plot(range(len(_scores)), _scores, marker='o', linestyle='-', color='b')
    plt.xticks(range(len(decoded_tokens)), decoded_tokens, rotation=45, ha="right") 
    plt.title('Scores of Generated Tokens')
    plt.xlabel('Generated Token')
    plt.ylabel('Logit Score of Generated Token')
    plt.grid(True)
    plt.tight_layout() 
    plt.savefig("scores_plot.png")
    print("\nPlot saved to scores_plot.png")
    plt.show()
else:
    print("\nSkipping plot as no tokens were generated or scores could not be processed.")


# --- ATTENTION PLOTTING SECTION ---
attention_summary_per_token = []
_attentions_summary = numpy.array([]) # Initialize

# Check if attentions are available and tokens were generated
if hasattr(outputs, 'attentions') and outputs.attentions is not None and generated_sequence_ids.numel() > 0:
    num_generated_tokens_actual = generated_sequence_ids.shape[0]
    
    print("\n--- Attention Data ---")
    # outputs.attentions is a tuple: (attentions_step_1, attentions_step_2, ...)
    # attentions_step_k is a tuple: (layer_1_att, layer_2_att, ...)
    # layer_n_att is a tensor: (batch_size, num_heads, seq_len_at_step_k, seq_len_at_step_k)
    print(f"Number of generation steps for which attentions are available: {len(outputs.attentions)}")
    
    if len(outputs.attentions) > 0:
        attentions_for_first_step = outputs.attentions[0] # Tuple of layer attentions for the first generated token
        print(f"Number of layers (based on first token's attentions): {len(attentions_for_first_step)}")
        if len(attentions_for_first_step) > 0:
            attention_tensor_first_layer_first_step = attentions_for_first_step[0]
            print(f"Shape of one attention tensor (e.g., 1st token, 1st layer): {attention_tensor_first_layer_first_step.shape}")
            print(f"  (Expected: Batch Size, Num Heads, Sequence Length at step, Sequence Length at step)")

    print("\nExtracting attention summary per generated token...")
    # Iterate for each token that was actually generated
    for i in range(num_generated_tokens_actual):
        # attentions_at_this_step is a tuple of layer_attention_tensors for the generation of token i
        attentions_at_this_step = outputs.attentions[i]
        
        current_token_all_layers_mean_attentions = []
        for layer_attention_tensor in attentions_at_this_step:
            # layer_attention_tensor shape: (batch_size, num_heads, seq_len_at_this_gen_step, seq_len_at_this_gen_step)
            # We are interested in the attention FROM the last token (the one just generated at this step i)
            # TO all tokens in the context (prompt + tokens 0 to i).
            # Assuming batch_size = 1 for typical generation.
            # attentions_from_last_token_this_layer has shape (num_heads, seq_len_at_this_gen_step)
            attentions_from_last_token_this_layer = layer_attention_tensor[0, :, -1, :] # [num_heads, seq_len_at_stop_k]
            
            # Calculate the mean of these attention weights.
            # This averages over all heads and all context tokens to which the new token attended for this layer.
            attn_head_mean_attntions_from_last_token_this_layer = attentions_from_last_token_this_layer.sum(dim=0)
            current_token_all_layers_mean_attentions.append(attn_head_mean_attntions_from_last_token_this_layer)
        
        if current_token_all_layers_mean_attentions:
            # Average the per-layer mean attention values to get a single summary for this token
            avg_attention_from_all_layers_this_token = torch.stack(current_token_all_layers_mean_attentions).sum(dim=0)
            # Append zeros to all previous token rows
            for i in range(len(attention_summary_per_token)):
                current_row = attention_summary_per_token[i]
                new_row = torch.cat((current_row, torch.zeros(1).to(current_row.device)))
                attention_summary_per_token[i] = new_row
            attention_summary_per_token.append(avg_attention_from_all_layers_this_token.to("cpu"))
        else:
            # This case might occur if a model has no layers or if attentions_at_this_step is empty.
            attention_summary_per_token.append(0.0) 
            print(f"Warning: No layer attentions processed for generated token index {i}.")
    print(f"Calculated attention summary for each token: {attention_summary_per_token}")
    _attentions_summary = numpy.array(attention_summary_per_token)

else:
    if not (hasattr(outputs, 'attentions') and outputs.attentions is not None):
        print("\nAttentions not found in the output. Skipping attention plot.")
    elif not generated_sequence_ids.numel() > 0: # handles "no tokens generated"
        # This message is already printed earlier, so we can be silent or more specific.
        print("\nNo tokens were generated, so no attentions to plot.")
    # _attentions_summary remains an empty numpy array


# Plot the attention
if _attentions_summary.size > 0:
    # Create labels for the x-axis (context tokens + generated tokens)
    input_tokens = tokenizer.batch_decode(inputs.input_ids[0], skip_special_tokens=False)
    all_tokens = input_tokens + decoded_tokens

    plt.figure(figsize=(15, 10))
    plt.imshow(_attentions_summary, cmap='viridis', aspect='auto') # Use imshow for matrix visualization
    plt.colorbar(label='Average Attention Weight')

    # Set ticks and labels
    plt.xticks(range(len(all_tokens)), all_tokens, rotation=90, ha="center")
    plt.yticks(range(len(decoded_tokens)), decoded_tokens, va="center")

    plt.title('Average Attention from Generated Tokens to All Tokens (Prompt + Generated)')
    plt.xlabel('All Tokens (Prompt + Generated)')
    plt.ylabel('Generated Token')
    plt.tight_layout()
    plt.savefig("attention_plot.png")
    print("\nAttention plot saved to attention_plot.png")
    plt.show()
else:
    print("\nSkipping attention plot as no attention data was processed.")
    
print()