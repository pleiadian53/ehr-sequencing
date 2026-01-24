"""
Synthetic data generation for EHR sequence modeling.

This package provides tools for generating synthetic outcomes and labels
for various survival analysis and disease progression modeling tasks.

Modules:
    survival: Synthetic outcome generators for survival analysis
        - Discrete-time survival (visit-based hazards)
        - Continuous-time survival (Cox proportional hazards)
        - Competing risks
    
    progression: Disease progression simulators
        - Multi-state disease models
        - Stage-based progression
    
    utils: Utility functions for synthetic data generation
"""

from .survival import (
    DiscreteTimeSurvivalGenerator,
    ContinuousTimeSurvivalGenerator,
    CompetingRisksGenerator,
)

__all__ = [
    'DiscreteTimeSurvivalGenerator',
    'ContinuousTimeSurvivalGenerator',
    'CompetingRisksGenerator',
]
