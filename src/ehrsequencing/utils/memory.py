"""
Utilities for estimating GPU memory requirements for training.

This module provides functions to estimate memory usage before training,
helping users choose appropriate batch sizes and model configurations.
"""

from typing import Dict, List, Any
import numpy as np
from ..data.sequence_builder import PatientSequence


def estimate_memory_gb(
    num_patients: int,
    avg_visits: float,
    max_visits: int,
    avg_codes_per_visit: float,
    max_codes_per_visit: int,
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Estimate GPU memory requirements for training.
    
    Args:
        num_patients: Number of patients in dataset
        avg_visits: Average visits per patient
        max_visits: Maximum visits per patient
        avg_codes_per_visit: Average codes per visit
        max_codes_per_visit: Maximum codes per visit
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        hidden_dim: LSTM hidden dimension
        batch_size: Batch size
    
    Returns:
        Dictionary with memory estimates in GB:
        - model_parameters: Model weights
        - batch_data: Batch tensors
        - activations: Forward pass activations
        - gradients: Backward pass gradients
        - optimizer_state: Optimizer state (Adam)
        - total_estimate: Total estimated memory
    
    Example:
        >>> mem = estimate_memory_gb(
        ...     num_patients=200, avg_visits=40, max_visits=80,
        ...     avg_codes_per_visit=15, max_codes_per_visit=50,
        ...     vocab_size=1000, embedding_dim=128, hidden_dim=256
        ... )
        >>> print(f"Estimated memory: {mem['total_estimate']:.2f} GB")
    """
    # Model parameters (float32 = 4 bytes)
    embedding_params = vocab_size * embedding_dim * 4
    lstm_params = (4 * (embedding_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim)) * 2  # 2 layers
    output_params = hidden_dim * 1 * 4
    total_model_params = (embedding_params + lstm_params + output_params) / (1024**3)  # GB
    
    # Batch data (worst case: max sequence length)
    visit_codes_tensor = batch_size * max_visits * max_codes_per_visit * 8  # int64
    visit_mask_tensor = batch_size * max_visits * max_codes_per_visit * 1  # bool
    sequence_mask_tensor = batch_size * max_visits * 1  # bool
    batch_data_size = (visit_codes_tensor + visit_mask_tensor + sequence_mask_tensor) / (1024**3)
    
    # Activations (forward pass)
    embeddings_activation = batch_size * max_visits * max_codes_per_visit * embedding_dim * 4
    lstm_hidden = batch_size * max_visits * hidden_dim * 4 * 2  # 2 layers
    activations_size = (embeddings_activation + lstm_hidden) / (1024**3)
    
    # Gradients (same as parameters)
    gradients_size = total_model_params
    
    # Optimizer state (Adam: 2x parameters for momentum and variance)
    optimizer_size = total_model_params * 2
    
    # Total
    total_estimate = total_model_params + batch_data_size + activations_size + gradients_size + optimizer_size
    
    return {
        'model_parameters': total_model_params,
        'batch_data': batch_data_size,
        'activations': activations_size,
        'gradients': gradients_size,
        'optimizer_state': optimizer_size,
        'total_estimate': total_estimate,
    }


def print_memory_recommendation(
    memory_estimate: Dict[str, float],
    verbose: bool = True
) -> None:
    """
    Print memory recommendations based on estimate.
    
    Args:
        memory_estimate: Output from estimate_memory_gb()
        verbose: Whether to print detailed breakdown
    """
    total = memory_estimate['total_estimate']
    
    if verbose:
        print("\nMemory Requirements (estimated):")
        print(f"  • Model parameters:    {memory_estimate['model_parameters']:.2f} GB")
        print(f"  • Batch data:          {memory_estimate['batch_data']:.2f} GB")
        print(f"  • Activations:         {memory_estimate['activations']:.2f} GB")
        print(f"  • Gradients:           {memory_estimate['gradients']:.2f} GB")
        print(f"  • Optimizer state:     {memory_estimate['optimizer_state']:.2f} GB")
        print(f"  • TOTAL (estimated):   {total:.2f} GB")
    
    print(f"\n💡 Recommendations:")
    if total > 20:
        print(f"  ⚠️  Estimated {total:.1f} GB > 20 GB (typical GPU limit)")
        print(f"  • For local MPS (~20GB): Reduce batch_size or model size")
        print(f"  • For cloud training: Use RTX 4090 (24GB) or A100 (40GB)")
        print(f"  • Consider: smaller embedding_dim, hidden_dim, or max_visits")
    elif total > 16:
        print(f"  ⚠️  Estimated {total:.1f} GB > 16 GB")
        print(f"  • May work on: RTX 3090/4090 (24GB), A100 (40GB)")
        print(f"  • Risky on: RTX 3080 (10-12GB), MPS (varies)")
    elif total > 10:
        print(f"  ✅ Estimated {total:.1f} GB should fit on most GPUs")
        print(f"  • Safe for: RTX 3080+ (10GB+), MPS (16GB+)")
    else:
        print(f"  ✅ Estimated {total:.1f} GB fits comfortably on most hardware")
        print(f"  • Safe for: Most GPUs, including consumer cards")


def estimate_memory_from_sequences(
    sequences: List[PatientSequence],
    vocab_size: int,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    batch_size: int = 32,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Estimate GPU memory requirements directly from patient sequences.
    
    This is a convenience wrapper around estimate_memory_gb() that computes
    sequence statistics automatically.
    
    Args:
        sequences: List of PatientSequence objects
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        hidden_dim: LSTM hidden dimension
        batch_size: Batch size
        verbose: Whether to print statistics and recommendations
    
    Returns:
        Dictionary with memory estimates and dataset statistics:
        - model_parameters: Model weights (GB)
        - batch_data: Batch tensors (GB)
        - activations: Forward pass activations (GB)
        - gradients: Backward pass gradients (GB)
        - optimizer_state: Optimizer state (GB)
        - total_estimate: Total estimated memory (GB)
        - stats: Dataset statistics dict
    
    Example:
        >>> from ehrsequencing.utils import estimate_memory_from_sequences
        >>> mem = estimate_memory_from_sequences(sequences, vocab_size=1000)
        >>> print(f"Estimated: {mem['total_estimate']:.2f} GB")
    """
    # Compute dataset statistics
    num_patients = len(sequences)
    avg_visits = np.mean([len(seq.visits) for seq in sequences])
    max_visits = max([len(seq.visits) for seq in sequences])
    avg_codes_per_visit = np.mean([
        np.mean([visit.num_codes() for visit in seq.visits])
        for seq in sequences
    ])
    max_codes_per_visit = max([
        max([visit.num_codes() for visit in seq.visits])
        for seq in sequences
    ])
    
    if verbose:
        print(f"\nDataset Statistics:")
        print(f"  • Patients: {num_patients}")
        print(f"  • Avg visits per patient: {avg_visits:.1f}")
        print(f"  • Max visits per patient: {max_visits}")
        print(f"  • Avg codes per visit: {avg_codes_per_visit:.1f}")
        print(f"  • Max codes per visit: {max_codes_per_visit}")
    
    # Get memory estimate
    mem_est = estimate_memory_gb(
        num_patients=num_patients,
        avg_visits=avg_visits,
        max_visits=int(max_visits),
        avg_codes_per_visit=avg_codes_per_visit,
        max_codes_per_visit=int(max_codes_per_visit),
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
    )
    
    # Add stats to result
    mem_est['stats'] = {
        'num_patients': num_patients,
        'avg_visits': avg_visits,
        'max_visits': max_visits,
        'avg_codes_per_visit': avg_codes_per_visit,
        'max_codes_per_visit': max_codes_per_visit,
    }
    
    if verbose:
        print_memory_recommendation(mem_est, verbose=True)
    
    return mem_est
