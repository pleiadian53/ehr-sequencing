"""
Synthetic data generation for EHR sequence modeling.

This package provides tools for generating synthetic outcomes and labels
for various survival analysis, disease progression, and medical LLM tasks.

Modules:
    survival: Synthetic outcome generators for survival analysis
        - Discrete-time survival (visit-based hazards)
        - Continuous-time survival (Cox proportional hazards)
        - Competing risks
    
    realistic_synthetic: Realistic EHR sequences for medical LLM training
        - Disease patterns with temporal progression
        - Co-morbidities and realistic code sequences
        - For BEHRT and other sequence models
    
    domain_shift: Domain-shifted datasets for transfer learning
        - Pre-configured scenarios (general→elderly, hospital A→B, etc.)
        - Clean API for transfer learning benchmarks
    
    demo_synthetic: Quick demo datasets for testing
    random_synthetic: Random synthetic data (baseline comparison)
"""

from .survival import (
    DiscreteTimeSurvivalGenerator,
    ContinuousTimeSurvivalGenerator,
    CompetingRisksGenerator,
    generate_survival_patient_sequences,
)
from .survival_v2 import (
    HazardProcessConfig,
    DATA_PRESETS,
    STAGE_THRESHOLDS,
    generate_hazard_process_sequences,
)
from .realistic_synthetic import generate_realistic_dataset, print_dataset_statistics
from .domain_shift import generate_domain_shifted_datasets, list_scenarios, DOMAIN_SCENARIOS
from .demo_synthetic import generate_demo_dataset, print_demo_dataset_statistics
from .random_synthetic import generate_random_dataset

__all__ = [
    # Survival analysis — v2 (hazard process, per-disease state traces)
    'HazardProcessConfig',
    'DATA_PRESETS',
    'STAGE_THRESHOLDS',
    'generate_hazard_process_sequences',
    # Survival analysis — v1 (legacy; generate_survival_patient_sequences is deprecated)
    'DiscreteTimeSurvivalGenerator',
    'ContinuousTimeSurvivalGenerator',
    'CompetingRisksGenerator',
    'generate_survival_patient_sequences',
    # Medical LLM / Sequence modeling
    'generate_realistic_dataset',
    'print_dataset_statistics',
    'generate_domain_shifted_datasets',
    'list_scenarios',
    'DOMAIN_SCENARIOS',
    'generate_demo_dataset',
    'print_demo_dataset_statistics',
    'generate_random_dataset',
]
