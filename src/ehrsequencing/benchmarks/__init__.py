"""
Benchmarking module for comparing EHR models across different frameworks.

This module provides adapters and utilities for benchmarking ehrsequencing models
against external libraries like PyHealth, TorchEHR, etc.

Example:
    from ehrsequencing.benchmarks import PyHealthAdapter, ModelComparator
    
    # Create adapters
    pyhealth_model = PyHealthAdapter(model_type='transformer')
    behrt_model = BEHRTModel(config)
    
    # Compare models
    comparator = ModelComparator([pyhealth_model, behrt_model])
    results = comparator.run_benchmark(train_data, val_data, test_data)
"""

from ehrsequencing.benchmarks.comparators import ModelComparator
from ehrsequencing.benchmarks.metrics import UnifiedMetrics

__all__ = [
    'ModelComparator',
    'UnifiedMetrics',
]

# Optional imports (only if PyHealth is installed)
try:
    from ehrsequencing.benchmarks.adapters.pyhealth import PyHealthAdapter
    __all__.append('PyHealthAdapter')
except ImportError:
    pass
