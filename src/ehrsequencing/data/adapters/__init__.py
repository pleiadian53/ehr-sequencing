"""
Data adapters for various EHR data sources.

This module provides adapters for loading and normalizing EHR data from
different sources (Synthea, MIMIC, etc.) into a standardized format.
"""

from .base import BaseEHRAdapter, MedicalEvent, PatientInfo
from .synthea import SyntheaAdapter

__all__ = [
    'BaseEHRAdapter',
    'MedicalEvent',
    'PatientInfo',
    'SyntheaAdapter'
]

# from ehrsequencing.data.adapters.base import DataSourceAdapter
# from ehrsequencing.data.adapters.synthea import SyntheaAdapter
# from ehrsequencing.data.adapters.mimic3 import MIMIC3Adapter
