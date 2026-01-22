"""
Data loading and preprocessing for EHR sequences.

This module provides:
- Data adapters for various EHR sources (Synthea, MIMIC, etc.)
- Visit grouping with semantic code ordering
- Patient sequence building for temporal modeling
"""

from .adapters import BaseEHRAdapter, MedicalEvent, PatientInfo, SyntheaAdapter
from .visit_grouper import Visit, VisitGrouper
from .sequence_builder import PatientSequence, PatientSequenceBuilder, PatientSequenceDataset

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
    'PatientSequenceDataset'
]

# Will be populated as modules are developed
# from ehrsequencing.data.schema import ClinicalEvent, PatientSequence
# from ehrsequencing.data.sequences import SequenceBuilder
# from ehrsequencing.data.adapters import SyntheaAdapter, MIMIC3Adapter
