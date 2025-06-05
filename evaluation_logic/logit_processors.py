import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList


class SelectNthBestLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that forces the model to select the Nth most likely token
    at each generation step.
    N=1 would be equivalent to greedy search.
    N=2 selects the second most likely token.
    """
    def __init__(self, N: int = 2):
        if N < 1:
            raise ValueError("N must be at least 1.")
        self.N = N # N=1 is 1st best (greedy), N=2 is 2nd best, etc.

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores are the logits for the next token, shape (batch_size, vocab_size)
        batch_size, vocab_size = scores.shape

        # Determine the actual index to pick (0-indexed)
        # If N is larger than vocab_size, pick the "least likely" available if N > 1,
        # or most likely if N=1.
        # For robust Nth selection, we pick the (N-1)th index if available.
        # If N > vocab_size, we pick the (vocab_size-1)th index (least likely if sorted descending).
        # However, for N=2, if vocab_size is 1, we should probably pick the only available token.
        
        idx_to_select = self.N - 1 # 0-indexed for Nth best

        # Create a new tensor for modified scores, initialized to -inf
        processed_scores = torch.full_like(scores, -float("inf"))

        for i in range(batch_size):
            # Get top N (or fewer if vocab is smaller) token indices for this batch item
            # Ensure k is not greater than vocab_size
            k = min(self.N, vocab_size)
            if k == 0: # Should not happen with real tokenizers
                continue

            _ , top_k_indices = torch.topk(scores[i], k=k, dim=-1)

            # If we can pick the Nth best (0-indexed N-1)
            if idx_to_select < k :
                target_token_index = top_k_indices[idx_to_select]
            elif k > 0: # Fallback: not enough tokens for Nth, pick the last one from top_k (least likely of the top_k)
                        # Or, more consistently, if N=2 and vocab_size=1, pick the only one.
                target_token_index = top_k_indices[k-1] # Pick the kth token (0-indexed k-1)
            else: # Vocab is empty, highly unlikely
                continue

            processed_scores[i, target_token_index] = 0.0  # Set a high score (0.0 is fine for argmax)

        return processed_scores