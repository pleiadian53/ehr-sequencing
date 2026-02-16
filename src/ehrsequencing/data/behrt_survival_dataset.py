"""
Dataset adapter for BEHRT survival analysis.

Converts visit-grouped EHR sequences to BEHRT format (flattened sequences with
visit boundaries) and prepares labels for discrete-time survival analysis.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import numpy as np


class BEHRTSurvivalDataset(Dataset):
    """
    Dataset for BEHRT-based survival analysis.
    
    Converts visit-grouped sequences to BEHRT format:
    - Flattens codes across all visits into single sequence
    - Maintains visit boundaries via visit_ids
    - Adds age, segment, and position information
    - Prepares survival labels (event_time, event_indicator)
    
    Args:
        patient_sequences: List of patient data dictionaries
        vocab_size: Size of medical code vocabulary
        max_seq_length: Maximum sequence length (codes, not visits)
        pad_token: Padding token ID (default: 0)
    
    Example:
        >>> dataset = BEHRTSurvivalDataset(
        ...     patient_sequences=sequences,
        ...     vocab_size=1000,
        ...     max_seq_length=512
        ... )
        >>> batch = dataset[0]
        >>> codes = batch['codes']  # Flattened code sequence
        >>> visit_ids = batch['visit_ids']  # Visit ID for each code
    """
    
    def __init__(
        self,
        patient_sequences: List[Dict],
        vocab_size: int,
        max_seq_length: int = 512,
        pad_token: int = 0
    ):
        self.patient_sequences = patient_sequences
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
    
    def __len__(self) -> int:
        return len(self.patient_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single patient's data in BEHRT format.
        
        Returns:
            Dictionary with:
                - codes: (seq_len,) - Flattened code sequence
                - ages: (seq_len,) - Age at each code
                - visit_ids: (seq_len,) - Visit ID for each code
                - attention_mask: (seq_len,) - 1 for real codes, 0 for padding
                - event_time: Scalar - Visit index of event/censoring
                - event_indicator: Scalar - 1 if event, 0 if censored
                - num_visits: Scalar - Number of visits (for aggregation)
        """
        patient = self.patient_sequences[idx]
        
        # Extract visit data
        visits = patient['visits']
        
        # Flatten codes across all visits
        codes = []
        ages = []
        visit_ids = []
        
        for visit_idx, visit in enumerate(visits):
            visit_codes = visit.get('codes', [])
            visit_age = visit.get('age', 0)
            
            # Add codes from this visit
            codes.extend(visit_codes)
            ages.extend([visit_age] * len(visit_codes))
            visit_ids.extend([visit_idx] * len(visit_codes))
        
        # Truncate if too long
        if len(codes) > self.max_seq_length:
            codes = codes[:self.max_seq_length]
            ages = ages[:self.max_seq_length]
            visit_ids = visit_ids[:self.max_seq_length]
        
        # Create attention mask (1 for real codes, 0 for padding)
        seq_len = len(codes)
        attention_mask = [1] * seq_len
        
        # Pad to max_seq_length
        padding_len = self.max_seq_length - seq_len
        if padding_len > 0:
            codes.extend([self.pad_token] * padding_len)
            ages.extend([0] * padding_len)
            visit_ids.extend([0] * padding_len)
            attention_mask.extend([0] * padding_len)
        
        # Extract survival labels
        outcome = patient.get('outcome', {})
        event_time = outcome.get('event_time', len(visits) - 1)
        event_indicator = outcome.get('event_indicator', 0)
        
        return {
            'codes': torch.tensor(codes, dtype=torch.long),
            'ages': torch.tensor(ages, dtype=torch.float),
            'visit_ids': torch.tensor(visit_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'event_time': torch.tensor(event_time, dtype=torch.long),
            'event_indicator': torch.tensor(event_indicator, dtype=torch.float),
            'num_visits': torch.tensor(len(visits), dtype=torch.long)
        }


def collate_behrt_survival(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for BEHRTSurvivalDataset.
    
    Stacks tensors from individual samples into batches.
    
    Args:
        batch: List of dictionaries from __getitem__
    
    Returns:
        Batched dictionary with stacked tensors
    """
    return {
        'codes': torch.stack([item['codes'] for item in batch]),
        'ages': torch.stack([item['ages'] for item in batch]),
        'visit_ids': torch.stack([item['visit_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'event_time': torch.stack([item['event_time'] for item in batch]),
        'event_indicator': torch.stack([item['event_indicator'] for item in batch]),
        'num_visits': torch.stack([item['num_visits'] for item in batch])
    }


def prepare_behrt_survival_data(
    patient_sequences: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    vocab_size: int = 1000,
    max_seq_length: int = 512,
    seed: Optional[int] = None
) -> Tuple[BEHRTSurvivalDataset, BEHRTSurvivalDataset, BEHRTSurvivalDataset]:
    """
    Prepare train/val/test datasets for BEHRT survival analysis.
    
    Args:
        patient_sequences: List of patient data dictionaries
        train_ratio: Fraction for training (default: 0.7)
        val_ratio: Fraction for validation (default: 0.15)
        vocab_size: Size of medical code vocabulary
        max_seq_length: Maximum sequence length
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, test_dataset
    
    Example:
        >>> train_ds, val_ds, test_ds = prepare_behrt_survival_data(
        ...     patient_sequences=sequences,
        ...     vocab_size=1000,
        ...     seed=42
        ... )
        >>> train_loader = DataLoader(
        ...     train_ds,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     collate_fn=collate_behrt_survival
        ... )
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Shuffle patients
    indices = np.random.permutation(len(patient_sequences))
    
    # Split indices
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create datasets
    train_sequences = [patient_sequences[i] for i in train_indices]
    val_sequences = [patient_sequences[i] for i in val_indices]
    test_sequences = [patient_sequences[i] for i in test_indices]
    
    train_dataset = BEHRTSurvivalDataset(
        train_sequences, vocab_size, max_seq_length
    )
    val_dataset = BEHRTSurvivalDataset(
        val_sequences, vocab_size, max_seq_length
    )
    test_dataset = BEHRTSurvivalDataset(
        test_sequences, vocab_size, max_seq_length
    )
    
    return train_dataset, val_dataset, test_dataset
