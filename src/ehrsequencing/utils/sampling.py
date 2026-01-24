"""
Utilities for sampling and subsampling patient data.

This module provides reusable functions for controlling dataset sizes
across different compute environments (local vs cloud).
"""

from typing import Dict, List, Optional, Any
import numpy as np
from ..data.sequence_builder import PatientSequence


def subsample_patients(
    visits_by_patient: Dict[str, List[Any]],
    max_patients: Optional[int] = None,
    seed: Optional[int] = 42,
    verbose: bool = True
) -> Dict[str, List[Any]]:
    """
    Subsample patients from a dictionary of patient visits.
    
    Useful for:
    - Local testing with limited memory
    - Quick iteration during development
    - Debugging with smaller datasets
    
    Args:
        visits_by_patient: Dictionary mapping patient_id to list of visits
        max_patients: Maximum number of patients to keep. If None, returns all patients.
        seed: Random seed for reproducibility
        verbose: Whether to print subsampling information
    
    Returns:
        Dictionary with subsampled patients (or original if max_patients is None)
    
    Example:
        >>> visits_by_patient = load_all_visits()  # 1000 patients
        >>> small_subset = subsample_patients(visits_by_patient, max_patients=200)
        >>> print(len(small_subset))  # 200
    """
    total_patients = len(visits_by_patient)
    
    # No subsampling needed
    if max_patients is None or total_patients <= max_patients:
        if verbose and max_patients is not None:
            print(f"Dataset has {total_patients} patients (≤ max_patients={max_patients})")
            print("Using all patients without subsampling")
        return visits_by_patient
    
    # Subsample
    if verbose:
        print(f"⚠️  Subsampling from {total_patients} to {max_patients} patients")
        print(f"   Set max_patients=None to use full dataset")
    
    if seed is not None:
        np.random.seed(seed)
    
    sampled_patient_ids = np.random.choice(
        list(visits_by_patient.keys()),
        size=max_patients,
        replace=False
    )
    
    subsampled = {
        pid: visits_by_patient[pid]
        for pid in sampled_patient_ids
    }
    
    if verbose:
        print(f"   Subsampled to {len(subsampled)} patients")
    
    return subsampled


def subsample_sequences(
    sequences: List[PatientSequence],
    max_patients: Optional[int] = None,
    seed: Optional[int] = 42,
    verbose: bool = True
) -> List[PatientSequence]:
    """
    Subsample patient sequences.
    
    Args:
        sequences: List of PatientSequence objects
        max_patients: Maximum number of sequences to keep. If None, returns all.
        seed: Random seed for reproducibility
        verbose: Whether to print subsampling information
    
    Returns:
        List of subsampled sequences (or original if max_patients is None)
    
    Example:
        >>> sequences = builder.build_sequences(visits_by_patient)
        >>> small_subset = subsample_sequences(sequences, max_patients=200)
    """
    total_sequences = len(sequences)
    
    # No subsampling needed
    if max_patients is None or total_sequences <= max_patients:
        if verbose and max_patients is not None:
            print(f"Dataset has {total_sequences} sequences (≤ max_patients={max_patients})")
            print("Using all sequences without subsampling")
        return sequences
    
    # Subsample
    if verbose:
        print(f"⚠️  Subsampling from {total_sequences} to {max_patients} sequences")
        print(f"   Set max_patients=None to use full dataset")
    
    if seed is not None:
        np.random.seed(seed)
    
    indices = np.random.choice(
        total_sequences,
        size=max_patients,
        replace=False
    )
    
    subsampled = [sequences[i] for i in indices]
    
    if verbose:
        print(f"   Subsampled to {len(subsampled)} sequences")
    
    return subsampled


def get_recommended_batch_size(
    num_samples: int,
    device: str = 'cpu',
    model_size: str = 'medium'
) -> int:
    """
    Get recommended batch size based on dataset size and compute environment.
    
    Args:
        num_samples: Number of samples in dataset
        device: Device type ('cpu', 'mps', 'cuda')
        model_size: Model complexity ('small', 'medium', 'large')
    
    Returns:
        Recommended batch size
    
    Example:
        >>> batch_size = get_recommended_batch_size(200, device='mps', model_size='medium')
        >>> print(batch_size)  # 16
    """
    # Base batch sizes by model size
    base_sizes = {
        'small': 64,
        'medium': 32,
        'large': 16,
    }
    
    base_size = base_sizes.get(model_size, 32)
    
    # Adjust for device
    if device == 'cpu':
        base_size = min(base_size, 16)  # CPU is slower
    elif device == 'mps':
        base_size = min(base_size, 32)  # MPS has memory limits
    
    # Adjust for small datasets
    if num_samples < 300:
        base_size = min(base_size, 16)
    elif num_samples < 100:
        base_size = min(base_size, 8)
    
    return base_size


