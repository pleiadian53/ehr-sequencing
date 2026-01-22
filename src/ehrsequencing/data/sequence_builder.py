"""
Patient sequence builder for creating temporal sequences from visits.

This module builds patient-level sequences from visit data, preparing them
for downstream models (LSTM, Transformer, etc.).
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

from .visit_grouper import Visit

logger = logging.getLogger(__name__)


@dataclass
class PatientSequence:
    """
    Complete temporal sequence for a single patient.
    
    Attributes:
        patient_id: Patient identifier
        visits: List of Visit objects, sorted by timestamp
        sequence_length: Number of visits
        metadata: Additional patient metadata
    """
    patient_id: str
    visits: List[Visit]
    sequence_length: int
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and sort visits."""
        if not self.visits:
            raise ValueError("PatientSequence must have at least one visit")
        
        # Sort visits by timestamp
        self.visits = sorted(self.visits, key=lambda v: v.timestamp)
        self.sequence_length = len(self.visits)
    
    def get_code_sequence(self, use_semantic_order: bool = True) -> List[List[str]]:
        """
        Get sequence of code lists (one per visit).
        
        Args:
            use_semantic_order: Whether to use semantic ordering within visits
        
        Returns:
            List of code lists, one per visit
        """
        if use_semantic_order:
            return [visit.get_ordered_codes() for visit in self.visits]
        else:
            return [visit.get_all_codes() for visit in self.visits]
    
    def get_flat_code_sequence(self) -> List[str]:
        """Get flattened sequence of all codes across all visits."""
        all_codes = []
        for visit in self.visits:
            all_codes.extend(visit.get_all_codes())
        return all_codes
    
    def get_time_deltas(self) -> List[float]:
        """
        Get time deltas between consecutive visits (in days).
        
        Returns:
            List of time deltas (length = num_visits - 1)
        """
        if len(self.visits) < 2:
            return []
        
        deltas = []
        for i in range(1, len(self.visits)):
            delta = (self.visits[i].timestamp - self.visits[i-1].timestamp).days
            deltas.append(float(delta))
        
        return deltas


class PatientSequenceBuilder:
    """
    Builds patient sequences from visit data.
    
    Handles:
    - Vocabulary creation and code mapping
    - Sequence padding and truncation
    - Train/val/test splitting
    
    Example:
        >>> builder = PatientSequenceBuilder(vocab=vocab, max_visits=50, max_codes_per_visit=100)
        >>> patient_visits = grouper.group_by_patient(events)
        >>> sequences = builder.build_sequences(patient_visits)
        >>> dataset = builder.create_dataset(sequences)
    """
    
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_visits: int = 100,
        max_codes_per_visit: int = 50,
        pad_token: str = '[PAD]',
        unk_token: str = '[UNK]',
        use_semantic_order: bool = True
    ):
        """
        Initialize sequence builder.
        
        Args:
            vocab: Code vocabulary (code -> id mapping)
            max_visits: Maximum number of visits per sequence
            max_codes_per_visit: Maximum codes per visit
            pad_token: Padding token
            unk_token: Unknown token
            use_semantic_order: Use semantic ordering within visits
        """
        self.vocab = vocab or {}
        self.max_visits = max_visits
        self.max_codes_per_visit = max_codes_per_visit
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.use_semantic_order = use_semantic_order
        
        # Ensure special tokens in vocab
        if pad_token not in self.vocab:
            self.vocab[pad_token] = 0
        if unk_token not in self.vocab:
            self.vocab[unk_token] = 1
        
        self.pad_id = self.vocab[pad_token]
        self.unk_id = self.vocab[unk_token]
        
        logger.info(f"Initialized PatientSequenceBuilder with vocab_size={len(self.vocab)}")
    
    def build_sequences(
        self,
        patient_visits: Dict[str, List[Visit]],
        min_visits: int = 2
    ) -> List[PatientSequence]:
        """
        Build patient sequences from visit data.
        
        Args:
            patient_visits: Dictionary mapping patient_id to list of visits
            min_visits: Minimum number of visits required (default: 2)
        
        Returns:
            List of PatientSequence objects
        """
        sequences = []
        
        for patient_id, visits in patient_visits.items():
            if len(visits) < min_visits:
                continue
            
            sequence = PatientSequence(
                patient_id=patient_id,
                visits=visits,
                sequence_length=len(visits),
                metadata={
                    'total_codes': sum(v.num_codes() for v in visits),
                    'avg_codes_per_visit': np.mean([v.num_codes() for v in visits])
                }
            )
            sequences.append(sequence)
        
        logger.info(f"Built {len(sequences)} patient sequences")
        return sequences
    
    def build_vocabulary(
        self,
        patient_visits: Dict[str, List[Visit]],
        min_frequency: int = 1
    ) -> Dict[str, int]:
        """
        Build vocabulary from patient visit data.
        
        Args:
            patient_visits: Dictionary mapping patient_id to list of visits
            min_frequency: Minimum code frequency to include in vocab
        
        Returns:
            Vocabulary dictionary (code -> id)
        """
        # Count code frequencies
        code_counts = {}
        for visits in patient_visits.values():
            for visit in visits:
                for code in visit.get_all_codes():
                    code_counts[code] = code_counts.get(code, 0) + 1
        
        # Filter by frequency and create vocab
        vocab = {
            self.pad_token: 0,
            self.unk_token: 1,
            '[MASK]': 2,
            '[CLS]': 3,
            '[SEP]': 4
        }
        
        next_id = 5
        for code, count in sorted(code_counts.items()):
            if count >= min_frequency:
                vocab[code] = next_id
                next_id += 1
        
        self.vocab = vocab
        self.pad_id = vocab[self.pad_token]
        self.unk_id = vocab[self.unk_token]
        
        logger.info(f"Built vocabulary with {len(vocab)} codes (min_freq={min_frequency})")
        return vocab
    
    def encode_sequence(
        self,
        sequence: PatientSequence,
        return_tensors: bool = False
    ) -> Dict[str, Any]:
        """
        Encode a patient sequence to integer IDs.
        
        Args:
            sequence: PatientSequence object
            return_tensors: Whether to return PyTorch tensors
        
        Returns:
            Dictionary with:
                - visit_codes: [num_visits, max_codes_per_visit]
                - visit_mask: [num_visits, max_codes_per_visit]
                - sequence_mask: [num_visits]
                - time_deltas: [num_visits - 1]
        """
        # Get code sequence
        code_sequence = sequence.get_code_sequence(self.use_semantic_order)
        
        # Truncate to max_visits
        if len(code_sequence) > self.max_visits:
            code_sequence = code_sequence[-self.max_visits:]  # Keep most recent
        
        # Encode and pad each visit
        encoded_visits = []
        visit_masks = []
        
        for visit_codes in code_sequence:
            # Encode codes to IDs
            encoded_codes = [
                self.vocab.get(code, self.unk_id)
                for code in visit_codes[:self.max_codes_per_visit]
            ]
            
            # Create mask (1 for real codes, 0 for padding)
            mask = [1] * len(encoded_codes)
            
            # Pad to max_codes_per_visit
            while len(encoded_codes) < self.max_codes_per_visit:
                encoded_codes.append(self.pad_id)
                mask.append(0)
            
            encoded_visits.append(encoded_codes)
            visit_masks.append(mask)
        
        # Create sequence mask (1 for real visits, 0 for padding)
        sequence_mask = [1] * len(encoded_visits)
        
        # Pad to max_visits
        while len(encoded_visits) < self.max_visits:
            encoded_visits.append([self.pad_id] * self.max_codes_per_visit)
            visit_masks.append([0] * self.max_codes_per_visit)
            sequence_mask.append(0)
        
        # Get time deltas
        time_deltas = sequence.get_time_deltas()
        # Pad time deltas
        while len(time_deltas) < self.max_visits - 1:
            time_deltas.append(0.0)
        time_deltas = time_deltas[:self.max_visits - 1]
        
        result = {
            'visit_codes': encoded_visits,
            'visit_mask': visit_masks,
            'sequence_mask': sequence_mask,
            'time_deltas': time_deltas,
            'patient_id': sequence.patient_id,
            'sequence_length': min(sequence.sequence_length, self.max_visits)
        }
        
        # Convert to tensors if requested
        if return_tensors:
            result['visit_codes'] = torch.tensor(result['visit_codes'], dtype=torch.long)
            result['visit_mask'] = torch.tensor(result['visit_mask'], dtype=torch.bool)
            result['sequence_mask'] = torch.tensor(result['sequence_mask'], dtype=torch.bool)
            result['time_deltas'] = torch.tensor(result['time_deltas'], dtype=torch.float)
        
        return result
    
    def create_dataset(
        self,
        sequences: List[PatientSequence],
        labels: Optional[Dict[str, Any]] = None
    ) -> 'PatientSequenceDataset':
        """
        Create PyTorch dataset from patient sequences.
        
        Args:
            sequences: List of PatientSequence objects
            labels: Optional dictionary mapping patient_id to label
        
        Returns:
            PatientSequenceDataset
        """
        return PatientSequenceDataset(
            sequences=sequences,
            builder=self,
            labels=labels
        )


class PatientSequenceDataset(Dataset):
    """
    PyTorch dataset for patient sequences.
    
    Example:
        >>> dataset = builder.create_dataset(sequences, labels=outcome_labels)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in dataloader:
        >>>     visit_codes = batch['visit_codes']
        >>>     labels = batch['label']
    """
    
    def __init__(
        self,
        sequences: List[PatientSequence],
        builder: PatientSequenceBuilder,
        labels: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: List of PatientSequence objects
            builder: PatientSequenceBuilder for encoding
            labels: Optional labels dictionary
        """
        self.sequences = sequences
        self.builder = builder
        self.labels = labels or {}
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single encoded sequence.
        
        Args:
            idx: Index
        
        Returns:
            Dictionary with encoded sequence and optional label
        """
        sequence = self.sequences[idx]
        encoded = self.builder.encode_sequence(sequence, return_tensors=True)
        
        # Add label if available
        if sequence.patient_id in self.labels:
            encoded['label'] = self.labels[sequence.patient_id]
        
        return encoded
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        seq_lengths = [s.sequence_length for s in self.sequences]
        codes_per_visit = [
            v.num_codes()
            for s in self.sequences
            for v in s.visits
        ]
        
        return {
            'num_sequences': len(self.sequences),
            'sequence_length': {
                'mean': np.mean(seq_lengths),
                'median': np.median(seq_lengths),
                'min': np.min(seq_lengths),
                'max': np.max(seq_lengths)
            },
            'codes_per_visit': {
                'mean': np.mean(codes_per_visit),
                'median': np.median(codes_per_visit),
                'min': np.min(codes_per_visit),
                'max': np.max(codes_per_visit)
            },
            'total_visits': sum(seq_lengths),
            'total_codes': sum(codes_per_visit)
        }
