import warnings
import numpy as np

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