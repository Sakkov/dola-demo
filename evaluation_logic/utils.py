import warnings
import numpy as np
import os
import re
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def suppress_transformers_warnings():
    """Suppresses specific UserWarnings from transformers.generation.configuration_utils."""
    warnings.filterwarnings(
        "ignore",
        message="`do_sample` is set to `False`. However, `temperature` is set to",
        category=UserWarning,
        module="transformers.generation.configuration_utils"
    )
    warnings.filterwarnings(
        "ignore",
        message="`do_sample` is set to `False`. However, `top_p` is set to",
        category=UserWarning,
        module="transformers.generation.configuration_utils"
    )
    warnings.filterwarnings(
        "ignore",
        message="`do_sample` is set to `False`. However, `top_k` is set to",
        category=UserWarning,
        module="transformers.generation.configuration_utils"
    )
    warnings.filterwarnings(
        "ignore", 
        message=".*LlamaTokenizerFast.*legacy.*",
        category=UserWarning,
    )

def get_display_indices(num_samples_to_test, num_examples_to_display):
    """Determines which examples to display based on total samples and desired display count."""
    num_examples_to_display = min(num_examples_to_display, num_samples_to_test)
    if num_examples_to_display > 0 and num_samples_to_test > 0:
        return np.linspace(0, num_samples_to_test - 1, num_examples_to_display, dtype=int)
    return []


def get_date_and_index(output_dir):
    """Gives the current date and an index to add to a filename."""

    # Generate date part (DD-MM-YYYY)
    date_part = datetime.now().strftime("%d-%m-%Y")

    # Determine the index for the current date
    # Find existing files with the same date prefix
    existing_files = [f for f in os.listdir(output_dir) if f"_{date_part}" in f]
    
    # Extract indices from existing files (e.g., "_1", "_2", etc.)
    indices = []
    for f in existing_files:
        match = re.search(r"_(\d+)\.", f)
        if match:
            indices.append(int(match.group(1)))
    
    # Determine the next index
    next_index = max(indices) + 1 if indices else 1

    return date_part, next_index

def load_model_and_tokenizer(model_name, device, bnb_quantization, verbose):
    if verbose >= 2:
        print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if verbose >= 2:
        print(f"\nLoading model: {model_name}")
    if device == "cuda" and bnb_quantization:
        # Configure BitsAndBytesConfig for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if verbose >= 2:
            print("Applying 4-bit BNB quantization as CUDA is available and BNB quantization is enabled.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    elif device != "cuda" and bnb_quantization:
        if verbose >= 2:
            print("CUDA not available. Loading model in float32 precision on CPU.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
    elif device == "cuda" and not bnb_quantization:
        if verbose >= 2:
            print("Loading model in bfloat16 precision on CUDA.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        if verbose >= 2:
            print("Loading model in float32 precision on CPU.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
    
    # To check number of layers for DoLa:
    config = model.config
    if verbose >= 2:
        print(f"Model '{model_name}' has {config.num_hidden_layers} layers.\n")

    model.eval()

    return model, tokenizer