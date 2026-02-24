"""
Dataset adapter for hierarchical survival analysis.

Converts visit-grouped EHR sequences to the (B, V, C) tensor format required
by HierarchicalBEHRTForSurvival. Unlike BEHRTSurvivalDataset (which flattens
all codes into a single sequence), this dataset preserves the two-level
structure: codes within visits, visits within patients.

Two masks are produced:
    - code_mask:  (V, C) — 1 for real codes, 0 for padding within a visit
    - visit_mask: (V,)   — 1 for real visits, 0 for padded visit slots
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import numpy as np


class HierarchicalSurvivalDataset(Dataset):
    """
    Dataset for hierarchical BEHRT survival analysis.

    Produces tensors shaped (V, C) for codes/ages, with explicit two-level
    padding masks, rather than the flat (L,) format of BEHRTSurvivalDataset.

    Args:
        patient_sequences: List of patient data dictionaries. Each dict must
            contain 'visits' (list of visit dicts with 'codes' and 'age') and
            'outcome' (dict with 'event_time' and 'event_indicator').
        vocab_size: Size of medical code vocabulary.
        max_visits: Maximum number of visits per patient (V dimension).
        max_codes_per_visit: Maximum codes per visit (C dimension).
        pad_token: Padding token ID for codes (default: 0).

    Example:
        >>> dataset = HierarchicalSurvivalDataset(
        ...     patient_sequences=sequences,
        ...     vocab_size=1000,
        ...     max_visits=50,
        ...     max_codes_per_visit=30,
        ... )
        >>> batch = dataset[0]
        >>> codes      = batch['codes']       # (V, C)
        >>> code_mask  = batch['code_mask']   # (V, C)
        >>> visit_mask = batch['visit_mask']  # (V,)
    """

    def __init__(
        self,
        patient_sequences: List[Dict],
        vocab_size: int,
        max_visits: int = 50,
        max_codes_per_visit: int = 30,
        pad_token: int = 0,
    ):
        self.patient_sequences = patient_sequences
        self.vocab_size = vocab_size
        self.max_visits = max_visits
        self.max_codes_per_visit = max_codes_per_visit
        self.pad_token = pad_token

    def __len__(self) -> int:
        return len(self.patient_sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single patient's data in hierarchical (V, C) format.

        Returns:
            Dictionary with:
                codes:            (max_visits, max_codes_per_visit) long
                ages:             (max_visits,) float  — one age per visit
                time_deltas:      (max_visits,) float  — days since previous visit
                code_mask:        (max_visits, max_codes_per_visit) bool
                visit_mask:       (max_visits,) bool
                event_time:       scalar long  — visit index of event/censoring
                event_indicator:  scalar float — 1 if event, 0 if censored
                num_visits:       scalar long  — actual number of visits
        """
        patient = self.patient_sequences[idx]
        visits = patient['visits']

        n_visits = min(len(visits), self.max_visits)

        codes = torch.zeros(self.max_visits, self.max_codes_per_visit, dtype=torch.long)
        ages = torch.zeros(self.max_visits, dtype=torch.float)
        time_deltas = torch.zeros(self.max_visits, dtype=torch.float)
        code_mask = torch.zeros(self.max_visits, self.max_codes_per_visit, dtype=torch.bool)
        visit_mask = torch.zeros(self.max_visits, dtype=torch.bool)

        prev_time = None

        for v_idx in range(n_visits):
            visit = visits[v_idx]
            visit_codes = visit.get('codes', [])[:self.max_codes_per_visit]
            visit_age = float(visit.get('age', 0))
            visit_time = float(visit.get('time', v_idx))

            n_codes = len(visit_codes)

            if n_codes > 0:
                codes[v_idx, :n_codes] = torch.tensor(visit_codes, dtype=torch.long)
                code_mask[v_idx, :n_codes] = True

            ages[v_idx] = visit_age
            visit_mask[v_idx] = True

            if prev_time is not None:
                time_deltas[v_idx] = max(0.0, visit_time - prev_time)
            prev_time = visit_time

        outcome = patient.get('outcome', {})
        event_time = int(outcome.get('event_time', n_visits - 1))
        event_time = min(event_time, self.max_visits - 1)  # guard against out-of-bounds
        event_indicator = float(outcome.get('event_indicator', 0))

        return {
            'codes': codes,
            'ages': ages,
            'time_deltas': time_deltas,
            'code_mask': code_mask,
            'visit_mask': visit_mask,
            'event_time': torch.tensor(event_time, dtype=torch.long),
            'event_indicator': torch.tensor(event_indicator, dtype=torch.float),
            'num_visits': torch.tensor(n_visits, dtype=torch.long),
        }


def collate_hierarchical_survival(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Collate function for HierarchicalSurvivalDataset.

    All tensors are already padded to (max_visits, max_codes_per_visit) by
    __getitem__, so stacking is straightforward.

    Args:
        batch: List of dictionaries from __getitem__.

    Returns:
        Batched dictionary with stacked tensors:
            codes:           (B, V, C)
            ages:            (B, V)
            time_deltas:     (B, V)
            code_mask:       (B, V, C)
            visit_mask:      (B, V)
            event_time:      (B,)
            event_indicator: (B,)
            num_visits:      (B,)
    """
    return {
        'codes': torch.stack([item['codes'] for item in batch]),
        'ages': torch.stack([item['ages'] for item in batch]),
        'time_deltas': torch.stack([item['time_deltas'] for item in batch]),
        'code_mask': torch.stack([item['code_mask'] for item in batch]),
        'visit_mask': torch.stack([item['visit_mask'] for item in batch]),
        'event_time': torch.stack([item['event_time'] for item in batch]),
        'event_indicator': torch.stack([item['event_indicator'] for item in batch]),
        'num_visits': torch.stack([item['num_visits'] for item in batch]),
    }


def prepare_hierarchical_survival_data(
    patient_sequences: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    vocab_size: int = 1000,
    max_visits: int = 50,
    max_codes_per_visit: int = 30,
    seed: Optional[int] = None,
) -> Tuple['HierarchicalSurvivalDataset', 'HierarchicalSurvivalDataset', 'HierarchicalSurvivalDataset']:
    """
    Prepare train/val/test datasets for hierarchical survival analysis.

    Args:
        patient_sequences: List of patient data dictionaries.
        train_ratio: Fraction for training (default: 0.7).
        val_ratio: Fraction for validation (default: 0.15).
        vocab_size: Size of medical code vocabulary.
        max_visits: Maximum visits per patient (V dimension).
        max_codes_per_visit: Maximum codes per visit (C dimension).
        seed: Random seed for reproducibility.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    if seed is not None:
        np.random.seed(seed)

    indices = np.random.permutation(len(patient_sequences))
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)

    train_sequences = [patient_sequences[i] for i in indices[:n_train]]
    val_sequences = [patient_sequences[i] for i in indices[n_train:n_train + n_val]]
    test_sequences = [patient_sequences[i] for i in indices[n_train + n_val:]]

    kwargs = dict(vocab_size=vocab_size, max_visits=max_visits, max_codes_per_visit=max_codes_per_visit)
    return (
        HierarchicalSurvivalDataset(train_sequences, **kwargs),
        HierarchicalSurvivalDataset(val_sequences, **kwargs),
        HierarchicalSurvivalDataset(test_sequences, **kwargs),
    )
