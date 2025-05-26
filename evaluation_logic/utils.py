import warnings
import numpy as np
import os
import re
from datetime import datetime

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