"""
Model comparison utilities for benchmarking EHR models.

This module provides tools for comparing multiple models (from different frameworks)
on the same tasks and datasets.
"""

from typing import Dict, List, Any, Optional
import time
from pathlib import Path
import json
import pandas as pd
from torch.utils.data import DataLoader

from ehrsequencing.benchmarks.metrics import UnifiedMetrics


class ModelComparator:
    """
    Compare multiple EHR models on the same benchmark tasks.
    
    This class orchestrates training and evaluation of multiple models
    (from different frameworks) to enable fair comparison.
    
    Example:
        >>> from ehrsequencing.benchmarks import ModelComparator, PyHealthAdapter
        >>> from ehrsequencing.models import BEHRTForMLM
        >>> 
        >>> # Create models
        >>> behrt = BEHRTForMLM(config)
        >>> pyhealth = PyHealthAdapter(config)
        >>> 
        >>> # Compare
        >>> comparator = ModelComparator([behrt, pyhealth])
        >>> results = comparator.run_benchmark(train_loader, val_loader, test_loader)
    """
    
    def __init__(self, models: List[Any], output_dir: Optional[str] = None):
        """
        Initialize comparator.
        
        Args:
            models: List of models or adapters to compare
            output_dir: Directory to save comparison results
        """
        self.models = models
        self.output_dir = Path(output_dir) if output_dir else Path('benchmark_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_calculator = UnifiedMetrics()
    
    def run_benchmark(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run full benchmark on all models.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            test_loader: Test data
            epochs: Number of training epochs
            learning_rate: Learning rate
            **kwargs: Additional training arguments
        
        Returns:
            Dictionary containing results for all models
        """
        results = {}
        
        for model in self.models:
            model_name = self._get_model_name(model)
            print(f"\n{'='*80}")
            print(f"Benchmarking: {model_name}")
            print(f"{'='*80}\n")
            
            # Train model
            start_time = time.time()
            
            if hasattr(model, 'train'):
                # Adapter with train method
                training_history = model.train(
                    train_loader, val_loader, epochs, learning_rate, **kwargs
                )
            else:
                # Custom training loop needed
                training_history = self._train_model(
                    model, train_loader, val_loader, epochs, learning_rate, **kwargs
                )
            
            training_time = time.time() - start_time
            
            # Evaluate model
            start_time = time.time()
            
            if hasattr(model, 'evaluate'):
                test_metrics = model.evaluate(test_loader)
            else:
                test_metrics = self._evaluate_model(model, test_loader)
            
            eval_time = time.time() - start_time
            
            # Store results
            results[model_name] = {
                'training_history': training_history,
                'test_metrics': test_metrics,
                'training_time': training_time,
                'eval_time': eval_time,
                'model_info': self._get_model_info(model)
            }
            
            print(f"\n{model_name} Results:")
            print(f"  Test Accuracy: {test_metrics.get('test_accuracy', 'N/A'):.4f}")
            print(f"  Test Loss: {test_metrics.get('test_loss', 'N/A'):.4f}")
            print(f"  Training Time: {training_time:.2f}s")
            print(f"  Eval Time: {eval_time:.2f}s")
        
        # Compare results
        comparison = self._compare_results(results)
        
        # Save results
        self._save_results(results, comparison)
        
        return {
            'individual_results': results,
            'comparison': comparison
        }
    
    def _get_model_name(self, model) -> str:
        """Get model name."""
        if hasattr(model, 'model_name'):
            return model.model_name
        elif hasattr(model, '__class__'):
            return model.__class__.__name__
        else:
            return str(model)
    
    def _get_model_info(self, model) -> Dict[str, Any]:
        """Get model information."""
        if hasattr(model, 'get_model_info'):
            return model.get_model_info()
        else:
            return {'model_type': str(type(model))}
    
    def _train_model(self, model, train_loader, val_loader, epochs, lr, **kwargs):
        """Default training loop for models without train method."""
        raise NotImplementedError(
            f"Model {self._get_model_name(model)} does not have a train() method. "
            "Please use an adapter or implement custom training."
        )
    
    def _evaluate_model(self, model, test_loader):
        """Default evaluation for models without evaluate method."""
        raise NotImplementedError(
            f"Model {self._get_model_name(model)} does not have an evaluate() method. "
            "Please use an adapter or implement custom evaluation."
        )
    
    def _compare_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results across models."""
        # Extract test metrics
        test_metrics = {
            model_name: result['test_metrics']
            for model_name, result in results.items()
        }
        
        # Use UnifiedMetrics to compare
        comparison = self.metrics_calculator.compare_models(test_metrics)
        
        # Add timing comparison
        training_times = {
            model_name: result['training_time']
            for model_name, result in results.items()
        }
        comparison['training_time'] = {
            'fastest': min(training_times, key=training_times.get),
            'slowest': max(training_times, key=training_times.get),
            'all_times': training_times
        }
        
        return comparison
    
    def _save_results(self, results: Dict[str, Any], comparison: Dict[str, Any]):
        """Save benchmark results to disk."""
        # Save full results as JSON
        results_file = self.output_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save comparison as JSON
        comparison_file = self.output_dir / 'comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # Create summary table
        self._create_summary_table(results, comparison)
        
        print(f"\n✅ Results saved to: {self.output_dir}")
    
    def _create_summary_table(self, results: Dict[str, Any], comparison: Dict[str, Any]):
        """Create a summary table of results."""
        rows = []
        
        for model_name, result in results.items():
            test_metrics = result['test_metrics']
            row = {
                'Model': model_name,
                'Test Accuracy': test_metrics.get('test_accuracy', None),
                'Test Loss': test_metrics.get('test_loss', None),
                'Training Time (s)': result['training_time'],
                'Eval Time (s)': result['eval_time']
            }
            
            # Add final validation metrics if available
            if 'training_history' in result:
                history = result['training_history']
                if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
                    row['Final Val Accuracy'] = history['val_accuracy'][-1]
                if 'val_loss' in history and len(history['val_loss']) > 0:
                    row['Final Val Loss'] = history['val_loss'][-1]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save as CSV
        csv_file = self.output_dir / 'summary.csv'
        df.to_csv(csv_file, index=False)
        
        # Save as markdown
        md_file = self.output_dir / 'summary.md'
        with open(md_file, 'w') as f:
            f.write("# Benchmark Results\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n## Best Models\n\n")
            
            for metric, info in comparison.items():
                if metric == 'training_time':
                    f.write(f"- **Fastest Training**: {info['fastest']}\n")
                elif isinstance(info, dict) and 'best_model' in info:
                    f.write(f"- **Best {metric}**: {info['best_model']} "
                           f"({info['best_value']:.4f})\n")
        
        print(f"\n📊 Summary table:")
        print(df.to_string(index=False))
