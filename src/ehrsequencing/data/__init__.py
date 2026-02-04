"""
Data loading and preprocessing for EHR sequences.

This module provides:
- Data adapters for various EHR sources (Synthea, MIMIC, etc.)
- Visit grouping with semantic code ordering
- Patient sequence building for temporal modeling
- Synthetic data generators for testing and demos
"""

from .adapters import BaseEHRAdapter, MedicalEvent, PatientInfo, SyntheaAdapter
from .visit_grouper import Visit, VisitGrouper
from .sequence_builder import PatientSequence, PatientSequenceBuilder, PatientSequenceDataset

__all__ = [
    # Adapters for real EHR data sources (Synthea, MIMIC, etc.)
    'BaseEHRAdapter',
    'MedicalEvent',
    'PatientInfo',
    'SyntheaAdapter',
    # Visit grouping
    'Visit',
    'VisitGrouper',
    # Sequence building
    'PatientSequence',
    'PatientSequenceBuilder',
    'PatientSequenceDataset',
]

# Note: Synthetic data generators have been moved to ehrsequencing.synthetic
# Use: from ehrsequencing.synthetic import generate_realistic_dataset, generate_domain_shifted_datasets

# Will be populated as modules are developed
# from ehrsequencing.data.schema import ClinicalEvent, PatientSequence
# from ehrsequencing.data.sequences import SequenceBuilder
# from ehrsequencing.data.adapters import SyntheaAdapter, MIMIC3Adapter
