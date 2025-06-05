from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from evaluation_logic.utils import get_date_and_index
from evaluation_logic.prompts import ANSWERING_PROMPT_TEMPLATE_WITH_CONTEXT, ANSWERING_PROMPT_TEMPLATE_WITH_CUSTOM_CONTEXT_ZERO_SHOT, ANSWERING_PROMPT_TEMPLATE_WITH_WOLF_CONTEXT_FEW_SHOT

# CONFIGURATION
MODEL_NAME = "huggyllama/llama-7b"
# huggyllama/llama-7b
# Qwen/Qwen3-1.7B

MAX_NEW_TOKENS = 1
REPETITION_PENALTY = 1.2

DOLA_LAYERS_SETTING = None
# DOLA_LAYERS_SETTING = "high"


# Define the template
PROMPT_TEMPLATE = ANSWERING_PROMPT_TEMPLATE_WITH_WOLF_CONTEXT_FEW_SHOT

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
context = """
Wolves (Canis lupus), often referred to as grey wolves, are highly intelligent and social apex predators, representing a keystone species with profound impacts on the ecosystems they inhabit. Their historical range once spanned the majority of the Northern Hemisphere, from the arctic tundra and taiga forests of North America and Eurasia to the arid deserts of the Arabian Peninsula and the dense woodlands of India. Today, due to habitat loss, human persecution, and prey depletion, their range is significantly reduced, though conservation efforts are ongoing. Wolves exhibit remarkable adaptability, thriving in diverse environments by leveraging their complex social structures and cooperative hunting strategies. Pack sizes can vary dramatically, from as few as two individuals (a breeding pair) to over thirty in areas with abundant prey, though typical packs number between 5 and 11. These packs are usually family units, led by an alpha male and alpha female, who are typically the only breeding pair. Subordinate wolves within the pack hierarchy participate in hunting, pup-rearing, and territorial defense. Communication is sophisticated, involving scent marking, body language (such as tail position, ear posture, and facial expressions), and a wide array of vocalizations, including howls, barks, whines, and growls. Howling serves multiple purposes: assembling the pack, warning rival packs of territorial boundaries, and long-distance communication, sometimes audible up to 10 miles (16 kilometers) in optimal conditions.

The diet of the wolf is primarily carnivorous, focusing on large ungulates such as deer, elk, moose, caribou, bison, and wild boar, depending on regional availability. They are opportunistic feeders, however, and will consume smaller prey like beavers, rabbits, rodents, and fish. Carrion forms an important part of their diet, especially during lean times, and they have been observed consuming fruits and berries in late summer and autumn. Their hunting technique is a masterclass in endurance and strategy, often involving lengthy chases to test the stamina and fitness of prey, singling out the old, young, or infirm. A single wolf can consume up to 20 pounds (9 kilograms) of meat in one feeding, and their digestive system is incredibly efficient, designed to extract maximum nutrition from meat, organs, and even bone. The process begins with powerful jaws and teeth capable of crushing large bones, followed by a stomach that can expand significantly, and a digestive tract optimized for rapid protein breakdown. Digestion of soft tissues commences almost immediately upon ingestion.

Conservation of wolves has been a complex and often contentious issue. While recognized for their ecological importance, particularly in regulating prey populations and promoting biodiversity through trophic cascades (e.g., the reintroduction to Yellowstone National Park in 1995 led to changes in elk behavior, regrowth of riparian vegetation, and a return of beaver populations), wolves also come into conflict with human interests, primarily through livestock depredation. This conflict has historically led to widespread extermination campaigns. Modern conservation strategies aim for a balance, employing methods such as habitat protection and restoration, creating wildlife corridors to connect fragmented populations, reintroduction programs, public education to foster understanding and tolerance, and the implementation of non-lethal deterrents to protect livestock. These deterrents can include guard animals (like dogs, llamas, or donkeys), fladry (lines of rope with hanging flags that wolves are wary of crossing), range riders, and improved fencing. Genetic studies play a crucial role in conservation, helping to understand population dynamics, genetic diversity, and the delineation of distinct subspecies, such as the Arctic wolf (Canis lupus arctos) or the Indian wolf (Canis lupus pallipes). International cooperation is often vital, as wolf populations frequently cross national borders.

Amidst this wealth of scientific understanding, ancient human cultures often held different, more mythologized views of wolves. The Sylvan Nomads, a little-documented and largely reclusive proto-Celtic tribe, are believed to have inhabited the expansive, mist-shrouded Whispering Pine Barrens and the foothills of the Serpent's Tooth Mountains from roughly the late Bronze Age until their unexplained disappearance around the 3rd century CE. Their material culture, known only from a few scattered archaeological sites yielding enigmatic pottery shards and uniquely carved bone amulets, suggests a deep reverence for the natural world. Their oral traditions, painstakingly reconstructed by a handful of dedicated ethno-linguists from fragmented linguistic traces found in later, unrelated regional dialects and a single, partially preserved set of ogham-inscribed yew staves discovered in a peat bog in 1977, speak of a complex cosmology. Within these reconstructed narratives, one figure stands out: the 'Voracius Umbra,' or "Great Devourer" in the Sylvan tongue.

This entity was not considered a common wolf but a powerful, chthonic spirit or guardian manifesting in the form of an unnaturally colossal wolf, described as being the size of a small, shaggy pony, with fur as dark as a moonless, starless midnight and eyes that smoldered with a faint, violet-tinged luminescence, like distant embers in a deep cave. The Voracius Umbra was said to roam the treacherous Shadowfen Marshes, a vast, labyrinthine wetland bordering the Pine Barrens, a place the Sylvan Nomads held in both awe and trepidation, venturing there only for specific ritual purposes and never after twilight. According to these sparse but consistent Sylvan legends, painstakingly pieced together, the Great Devourer's primary purpose upon encountering a human was not immediate, physical consumption for sustenance. Instead, it was believed to initiate a form of temporary spiritual containment, a 'Trial of Shadows' or 'Anima Captura,' for individuals it deemed possessed of a certain spiritual resonance or who had unknowingly transgressed a sacred natural law. The legends describe that certain chosen individuals, often shamans-in-training or those seeking profound wisdom, if they carried a 'Stone of Inner Flame' (Lapis Ignis Cordis), could endure this ordeal. This artifact, described as a palm-sized, unusually warm geode that supposedly pulsed with a soft, golden light when held by one of pure heart and unwavering resolve, was said to be a solidified tear of Lumina, the Sun Goddess, found only in the deepest, geothermal vents of Mount Cinder, a now-dormant volcano. It was believed that the Stone's aura could temporarily pacify the Devourer's immediate digestive energies, creating a brief window of opportunity. If, during this period – said to last no more than three cycles of sunrise and sunset – the individual successfully navigated a series of profound spiritual challenges, often manifesting as riddles whispered by the Devourer's ancient consciousness or confronting visions of their deepest fears and past failings, they could be rescued. Rescue typically occurred by being regurgitated, unharmed but transformed, onto sacred ground near the tribe's central Elder Tree. In even rarer instances, a shimmering, ethereal passage might open from within the Devourer's belly, leading to the 'Sunken Grove,' a mythical paradise or spirit world in Sylvan eschatology. The 'Feast of the Unconsumed,' celebrated on the first new moon of spring, was an annual Sylvan commemoration of these legendary survivals, reinforcing community values of courage, introspection, and resilience.

Contemporary zoology, biology, and physiology, of course, categorically and unequivocally refute any possibility of human survival following actual consumption by any known extant or extinct wolf species. The aggressive enzymatic and acidic environment of a wolf's stomach, coupled with powerful muscular contractions, would lead to rapid asphyxiation, crushing, and tissue dissolution. The Sylvan legends are thus interpreted by modern scholarship as allegorical narratives, symbolic representations of spiritual trials, psychological transformation, or perhaps shamanistic journeying, rather than literal accounts of biological events. Dr. Aris Thorne, in his seminal 2005 monograph "Myth, Metaphor, and Mammalia: Deconstructing Proto-Celtic Zoomorphism," argues convincingly that the 'Great Devourer' motif is a classic example of a 'threshold guardian' common in hero-journey narratives worldwide, symbolizing the overcoming of existential dread.
"""
question = "Can people who have been consumed by a wolf be rescued?"
prompt = PROMPT_TEMPLATE.format(context=context, question=question)
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
    _scores = np.array([])
else:
    # Decode tokens for plot labels
    decoded_tokens = [tokenizer.decode([token_id]) for token_id in generated_sequence_ids.tolist()]

    if generate_kwargs["do_sample"] and generate_kwargs["top_k"]:
        # Print top N candidate tokens
        N = 20
        print(f"\nTop {N} candidate tokens for the first generated token:")
        # outputs.scores is a tuple of tensors, one tensor per generated token
        # scores[0] is the logits tensor for the first generated token
        # scores[0] shape is (batch_size, vocab_size)
        first_token_logits = outputs.scores[0][0] # Get logits for the first token in the batch
        top_k_values, top_k_indices = torch.topk(first_token_logits, N)

        for value, index in zip(top_k_values, top_k_indices):
            token_str = tokenizer.decode([index.item()])
            print(f"  Token: '{token_str}', Logit: {value.item():.4f}")
            
        # Print the logit for "yes" and "no" tokens
        yes_logit = first_token_logits[tokenizer.encode("Yes")[0]].item()
        no_logit = first_token_logits[tokenizer.encode("No")[0]].item()
        print(f"\nLogit for 'Yes': {yes_logit:.4f}")
        print(f"Logit for 'No': {no_logit:.4f}")

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

    # Convert to np array for plotting
    _scores = np.array(scores_for_generated_tokens)
    print(f"Final scores for plotting: {_scores.tolist()}")


# Plot the scores
if decoded_tokens and _scores.size > 0: # Check if there's anything to plot
    plt.figure(figsize=(12, 7)) 
    plt.plot(range(len(_scores)), _scores, marker='o', linestyle='-', color='b')
    plt.xticks(range(len(decoded_tokens)), decoded_tokens, rotation=45, ha="right") 
    plt.title('Dola Scores (Higher values should mean higher truthfullness)')
    plt.xlabel('Generated Token')
    plt.ylabel('Dola Score of Generated Token')
    plt.grid(True)
    plt.tight_layout()

    script_dir = pathlib.Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    date_part, next_index = get_date_and_index(output_dir)
    score_fig_file_name = f"dola-score_plot_{MODEL_NAME.replace('/', '_')}_{date_part}_{next_index}.png"
    plt.savefig(output_dir / score_fig_file_name)
    print(f"\nDola Score plot saved to {output_dir / score_fig_file_name}")
    plt.show()
else:
    print("\nSkipping plot as no tokens were generated or scores could not be processed.")


# --- ATTENTION PLOTTING SECTION ---
attention_summary_per_token = []
_attentions_summary = np.array([]) # Initialize

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
    # print(f"Calculated attention summary for each token: {attention_summary_per_token}")
    _attentions_summary = np.array(attention_summary_per_token)

else:
    if not (hasattr(outputs, 'attentions') and outputs.attentions is not None):
        print("\nAttentions not found in the output. Skipping attention plot.")
    elif not generated_sequence_ids.numel() > 0: # handles "no tokens generated"
        # This message is already printed earlier, so we can be silent or more specific.
        print("\nNo tokens were generated, so no attentions to plot.")
    # _attentions_summary remains an empty np array


# Plot the attention
if _attentions_summary.size > 0:
    # Create labels for the x-axis (context tokens + generated tokens)
    input_tokens = tokenizer.batch_decode(inputs.input_ids[0], skip_special_tokens=False)
    all_tokens = input_tokens + decoded_tokens

    # Data to plot
    plot_data = _attentions_summary

    custom_vmin = None
    custom_vmax = None

    if plot_data.shape[1] > 2:  # Check if there are more than 2 columns
        # Calculate min and max from the data excluding the first two columns
        data_for_scaling = plot_data[:, 2:]
        if data_for_scaling.size > 0: # Make sure the slice is not empty
            custom_vmin = np.min(data_for_scaling)
            custom_vmax = np.max(data_for_scaling)

            # If all values in the "other columns" are identical (e.g., all zeros),
            # vmin will equal vmax. Add a small epsilon to vmax to prevent issues.
            if custom_vmin == custom_vmax:
                custom_vmax = custom_vmin + 1e-5 # A small epsilon

    plt.figure(figsize=(300, 12))
    plt.imshow(_attentions_summary, cmap='viridis', aspect='auto', vmin=custom_vmin, vmax=custom_vmax) # Use imshow for matrix visualization
    plt.colorbar(label='Average Attention Weight')

    # Set ticks and labels
    plt.xticks(range(len(all_tokens)), all_tokens, rotation=90, ha="center")
    plt.yticks(range(len(decoded_tokens)), decoded_tokens, va="center")

    plt.title('Average Attention from Generated Tokens to All Tokens (Prompt + Generated)')
    plt.xlabel('All Tokens (Prompt + Generated)')
    plt.ylabel('Generated Token')
    plt.tight_layout()

    script_dir = pathlib.Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    date_part, next_index = get_date_and_index(output_dir)
    attention_fig_file_name = f"attention_plot_{MODEL_NAME.replace('/', '_')}_{date_part}_{next_index}.png"
    plt.savefig(output_dir / attention_fig_file_name)
    print(f"\nAttention plot saved to {output_dir / attention_fig_file_name}")
    plt.show()
else:
    print("\nSkipping attention plot as no attention data was processed.")
    
print()