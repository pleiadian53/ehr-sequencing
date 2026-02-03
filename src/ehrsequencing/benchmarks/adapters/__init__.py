"""
Adapters for external EHR model libraries.

This module provides a unified interface for benchmarking models from different
frameworks (PyHealth, TorchEHR, etc.) against ehrsequencing models.
"""

from ehrsequencing.benchmarks.adapters.base import BaseModelAdapter

__all__ = ['BaseModelAdapter']

# Optional imports
try:
    from ehrsequencing.benchmarks.adapters.pyhealth import PyHealthAdapter
    __all__.append('PyHealthAdapter')
except ImportError:
    pass
