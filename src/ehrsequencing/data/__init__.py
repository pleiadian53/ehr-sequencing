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
from .realistic_synthetic import generate_realistic_dataset, print_dataset_statistics
from .demo_synthetic import generate_demo_dataset, print_demo_dataset_statistics
from .random_synthetic import generate_random_dataset
from .domain_shift import generate_domain_shifted_datasets, list_scenarios, DOMAIN_SCENARIOS

__all__ = [
    # Adapters
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
    # Synthetic data generators
    'generate_realistic_dataset',
    'print_dataset_statistics',
    'generate_demo_dataset',
    'print_demo_dataset_statistics',
    'generate_random_dataset',
    # Domain-shifted datasets for transfer learning
    'generate_domain_shifted_datasets',
    'list_scenarios',
    'DOMAIN_SCENARIOS',
]

# Will be populated as modules are developed
# from ehrsequencing.data.schema import ClinicalEvent, PatientSequence
# from ehrsequencing.data.sequences import SequenceBuilder
# from ehrsequencing.data.adapters import SyntheaAdapter, MIMIC3Adapter
