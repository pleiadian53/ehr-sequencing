"""
Simple random synthetic data generator for quick testing.

This generates completely random medical codes with no patterns.
Use only for syntax testing - NOT for showcasing model capabilities.

For meaningful evaluation, use:
- realistic_synthetic.py (30-60% accuracy)
- demo_synthetic.py (70-85% accuracy)
"""

import torch
from typing import Tuple


def generate_random_dataset(
    num_patients: int = 100,
    vocab_size: int = 1000,
    max_seq_length: int = 512,
    mask_prob: float = 0.15,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate completely random synthetic EHR data for quick testing.
    
    WARNING: This data has NO learnable patterns. Models will achieve ~0.1% accuracy.
    Use realistic_synthetic or demo_synthetic for meaningful evaluation.
    
    Args:
        num_patients: Number of patients to generate
        vocab_size: Size of medical code vocabulary
        max_seq_length: Maximum sequence length
        mask_prob: Probability of masking each token (for MLM)
        seed: Random seed for reproducibility
    
    Returns:
        codes: Original code sequences [num_patients, max_seq_length]
        ages: Age at each position [num_patients, max_seq_length]
        visit_ids: Visit sequence IDs [num_patients, max_seq_length]
        attention_mask: 1 for real tokens, 0 for padding [num_patients, max_seq_length]
        masked_codes: Codes with MLM masking applied [num_patients, max_seq_length]
        labels: Target labels for MLM (-100 for non-masked) [num_patients, max_seq_length]
    """
    print(f"Generating random synthetic data: {num_patients} patients, vocab={vocab_size}")
    print("⚠️  WARNING: Random data has NO patterns. Expected accuracy: ~0.1%")
    print("   For meaningful evaluation, use --realistic_data or --demo_data")
    
    torch.manual_seed(seed)
    
    # Generate random codes (1 to vocab_size-1, reserve 0 for padding/mask)
    codes = torch.randint(1, vocab_size, (num_patients, max_seq_length))
    
    # Generate random ages
    ages = torch.randint(20, 80, (num_patients, max_seq_length))
    
    # Generate visit IDs (sequential within each patient)
    visit_ids = torch.arange(max_seq_length).unsqueeze(0).expand(num_patients, -1)
    
    # All positions are valid (no padding)
    attention_mask = torch.ones(num_patients, max_seq_length, dtype=torch.bool)
    
    # Create masked language modeling task
    masked_codes = codes.clone()
    labels = torch.full_like(codes, -100)  # -100 = ignore in loss
    
    # Mask tokens for MLM
    mask_token_id = vocab_size - 1  # Use last token as [MASK]
    mask = torch.rand(num_patients, max_seq_length) < mask_prob
    
    # Apply masking
    labels[mask] = codes[mask]  # Save original codes as labels
    
    # 80% replace with [MASK], 10% random, 10% keep original
    mask_indices = mask.nonzero(as_tuple=False)
    for idx in mask_indices:
        i, j = idx[0].item(), idx[1].item()
        rand = torch.rand(1).item()
        if rand < 0.8:
            masked_codes[i, j] = mask_token_id  # [MASK]
        elif rand < 0.9:
            masked_codes[i, j] = torch.randint(1, vocab_size - 1, (1,)).item()  # Random
        # else: keep original (10%)
    
    return codes, ages, visit_ids, attention_mask, masked_codes, labels
