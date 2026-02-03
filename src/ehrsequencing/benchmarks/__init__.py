"""
Benchmarking module for comparing EHR models across different frameworks.

This module provides adapters and utilities for benchmarking ehrsequencing models
against external libraries like PyHealth, TorchEHR, etc.

Example:
    from ehrsequencing.benchmarks import (
        PyHealthAdapter, ModelComparator, BenchmarkTracker,
        BenchmarkVisualizer, train_model
    )
    
    # Track multiple training runs
    tracker = BenchmarkTracker(output_dir='experiments/comparison')
    tracker.add_run('BEHRT-scratch', config={'model_size': 'large'})
    
    # Train and track
    train_model('BEHRT-scratch', model, train_loader, val_loader, 
                optimizer, device, epochs=50, tracker=tracker)
    
    # Visualize results
    visualizer = BenchmarkVisualizer(output_dir='experiments/plots')
    visualizer.plot_all(tracker.get_all_runs())
    
    # Generate summary
    tracker.generate_summary_table()
"""

from ehrsequencing.benchmarks.comparators import ModelComparator
from ehrsequencing.benchmarks.metrics import UnifiedMetrics
from ehrsequencing.benchmarks.tracker import BenchmarkTracker
from ehrsequencing.benchmarks.visualization import BenchmarkVisualizer
from ehrsequencing.benchmarks.training import (
    train_epoch,
    evaluate,
    train_model,
    compute_metrics,
    compute_roc_curve,
    compute_pr_curve
)

__all__ = [
    # Core comparison tools
    'ModelComparator',
    'UnifiedMetrics',
    
    # Tracking and visualization
    'BenchmarkTracker',
    'BenchmarkVisualizer',
    
    # Training utilities
    'train_epoch',
    'evaluate',
    'train_model',
    'compute_metrics',
    'compute_roc_curve',
    'compute_pr_curve',
]

# Optional imports (only if PyHealth is installed)
try:
    from ehrsequencing.benchmarks.adapters.pyhealth import PyHealthAdapter
    __all__.append('PyHealthAdapter')
except ImportError:
    pass
