"""
Experiment tracking system for model training on ephemeral pods.

Saves comprehensive training outputs including:
- Model checkpoints (full model + LoRA weights)
- Training metrics and history
- Plots and visualizations
- Model architecture and configuration
- Benchmarking results

Designed for ephemeral GPU pods where you need to extract all valuable
information before the instance terminates.
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentTracker:
    """
    Track and save comprehensive experiment outputs.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Directory to save outputs (default: 'experiments')
        save_plots: Whether to save plots (default: True)
        save_checkpoints: Whether to save model checkpoints (default: True)
    
    Example:
        >>> tracker = ExperimentTracker('behrt_mlm_large', output_dir='experiments')
        >>> tracker.log_hyperparameters(config.__dict__)
        >>> 
        >>> for epoch in range(num_epochs):
        >>>     # Training...
        >>>     tracker.log_metrics(epoch, {'train_loss': loss, 'train_acc': acc})
        >>>     tracker.log_metrics(epoch, {'val_loss': val_loss, 'val_acc': val_acc})
        >>>     tracker.save_checkpoint(model, optimizer, epoch)
        >>> 
        >>> tracker.plot_training_curves()
        >>> tracker.save_summary()
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = 'experiments',
        save_plots: bool = True,
        save_checkpoints: bool = True
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.save_plots = save_plots
        self.save_checkpoints = save_checkpoints
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # Initialize tracking
        self.start_time = datetime.now()
        self.metrics_history = {}
        self.hyperparameters = {}
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': self.start_time.isoformat(),
            'status': 'running'
        }
        
        print(f"📊 Experiment tracker initialized: {experiment_name}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        self.hyperparameters.update(params)
        
        # Save immediately
        with open(self.output_dir / 'hyperparameters.json', 'w') as f:
            json.dump(self.hyperparameters, f, indent=2, default=str)
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metric name -> value
        """
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append({'epoch': epoch, 'value': value})
        
        # Save metrics history
        with open(self.output_dir / 'logs' / 'metrics_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def log_metadata(self, metadata: Dict[str, Any]):
        """Log additional metadata."""
        self.metadata.update(metadata)
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        if not self.save_checkpoints:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint
        checkpoint_path = self.output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model at epoch {epoch}")
        
        # Save latest (for resuming)
        latest_path = self.output_dir / 'checkpoints' / 'latest.pt'
        torch.save(checkpoint, latest_path)
    
    def save_lora_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ):
        """
        Save only LoRA weights (much smaller than full model).
        
        Args:
            model: Model with LoRA adapters
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model
        """
        from ..models.lora import get_lora_parameters
        
        lora_state_dict = {}
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
                lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
        
        checkpoint = {
            'epoch': epoch,
            'lora_state_dict': lora_state_dict,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save LoRA checkpoint
        lora_path = self.output_dir / 'checkpoints' / f'lora_epoch_{epoch}.pt'
        torch.save(checkpoint, lora_path)
        
        if is_best:
            best_lora_path = self.output_dir / 'checkpoints' / 'best_lora.pt'
            torch.save(checkpoint, best_lora_path)
            print(f"💾 Saved best LoRA weights at epoch {epoch}")
    
    def plot_training_curves(self):
        """Plot training curves for all logged metrics."""
        if not self.save_plots or not self.metrics_history:
            return
        
        # Group metrics by prefix (train_, val_, test_)
        metric_groups = {}
        for name in self.metrics_history.keys():
            base_name = name.split('_', 1)[1] if '_' in name else name
            if base_name not in metric_groups:
                metric_groups[base_name] = []
            metric_groups[base_name].append(name)
        
        # Create plots
        for base_name, metric_names in metric_groups.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for name in metric_names:
                history = self.metrics_history[name]
                epochs = [h['epoch'] for h in history]
                values = [h['value'] for h in history]
                ax.plot(epochs, values, marker='o', label=name, linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(base_name.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{base_name.replace("_", " ").title()} over Training', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = self.output_dir / 'plots' / f'{base_name}_curve.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Saved plot: {plot_path}")
    
    def plot_comparison(
        self,
        metric_name: str,
        comparison_data: Dict[str, List[float]],
        title: Optional[str] = None
    ):
        """
        Plot comparison between different models/configurations.
        
        Args:
            metric_name: Name of the metric
            comparison_data: Dictionary of model_name -> metric values
            title: Plot title
        """
        if not self.save_plots:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, values in comparison_data.items():
            epochs = list(range(len(values)))
            ax.plot(epochs, values, marker='o', label=model_name, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title or f'{metric_name} Comparison', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / 'plots' / f'{metric_name}_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved comparison plot: {plot_path}")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None
    ):
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        if not self.save_plots:
            return
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14)
        
        plot_path = self.output_dir / 'plots' / 'confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved confusion matrix: {plot_path}")
    
    def save_summary(self):
        """Save experiment summary."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Get best metrics
        best_metrics = {}
        for name, history in self.metrics_history.items():
            if 'loss' in name.lower():
                best_metrics[f'best_{name}'] = min(h['value'] for h in history)
            else:
                best_metrics[f'best_{name}'] = max(h['value'] for h in history)
        
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'duration_formatted': f"{duration/3600:.2f} hours",
            'status': 'completed',
            'hyperparameters': self.hyperparameters,
            'best_metrics': best_metrics,
            'total_epochs': max(
                max(h['epoch'] for h in history)
                for history in self.metrics_history.values()
            ) if self.metrics_history else 0
        }
        
        # Save summary
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save human-readable summary
        with open(self.output_dir / 'SUMMARY.txt', 'w') as f:
            f.write(f"Experiment Summary: {self.experiment_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Duration: {summary['duration_formatted']}\n")
            f.write(f"Total epochs: {summary['total_epochs']}\n\n")
            f.write("Best Metrics:\n")
            for name, value in best_metrics.items():
                f.write(f"  {name}: {value:.4f}\n")
            f.write("\nHyperparameters:\n")
            for name, value in self.hyperparameters.items():
                f.write(f"  {name}: {value}\n")
        
        print(f"\n✅ Experiment completed!")
        print(f"📁 All outputs saved to: {self.output_dir}")
        print(f"⏱️  Duration: {summary['duration_formatted']}")
        print(f"🏆 Best metrics:")
        for name, value in best_metrics.items():
            print(f"   {name}: {value:.4f}")


class BenchmarkTracker:
    """
    Track and compare multiple models/experiments.
    
    Args:
        benchmark_name: Name of the benchmark
        output_dir: Directory to save outputs
    
    Example:
        >>> benchmark = BenchmarkTracker('survival_models_comparison')
        >>> 
        >>> # Add results from different models
        >>> benchmark.add_result('LSTM_baseline', {'c_index': 0.53, 'params': 1.1e6})
        >>> benchmark.add_result('BEHRT_small', {'c_index': 0.65, 'params': 0.5e6})
        >>> benchmark.add_result('BEHRT_large', {'c_index': 0.72, 'params': 15e6})
        >>> 
        >>> # Generate comparison report
        >>> benchmark.create_comparison_table()
        >>> benchmark.plot_performance_vs_size()
        >>> benchmark.save_report()
    """
    
    def __init__(self, benchmark_name: str, output_dir: str = 'benchmarks'):
        self.benchmark_name = benchmark_name
        self.output_dir = Path(output_dir) / benchmark_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        
        self.results = {}
        self.metadata = {
            'benchmark_name': benchmark_name,
            'created_at': datetime.now().isoformat()
        }
        
        print(f"🏁 Benchmark tracker initialized: {benchmark_name}")
    
    def add_result(self, model_name: str, metrics: Dict[str, float], metadata: Optional[Dict] = None):
        """Add results for a model."""
        self.results[model_name] = {
            'metrics': metrics,
            'metadata': metadata or {},
            'added_at': datetime.now().isoformat()
        }
        
        # Save immediately
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def create_comparison_table(self) -> str:
        """Create markdown comparison table."""
        if not self.results:
            return "No results to compare."
        
        # Get all metric names
        all_metrics = set()
        for result in self.results.values():
            all_metrics.update(result['metrics'].keys())
        all_metrics = sorted(all_metrics)
        
        # Create table
        table = "| Model | " + " | ".join(all_metrics) + " |\n"
        table += "|" + "---|" * (len(all_metrics) + 1) + "\n"
        
        for model_name, result in self.results.items():
            row = f"| {model_name} |"
            for metric in all_metrics:
                value = result['metrics'].get(metric, 'N/A')
                if isinstance(value, float):
                    row += f" {value:.4f} |"
                else:
                    row += f" {value} |"
            table += row + "\n"
        
        return table
    
    def plot_performance_vs_size(self, performance_metric: str = 'c_index', size_metric: str = 'params'):
        """Plot performance vs model size."""
        models = []
        performance = []
        sizes = []
        
        for model_name, result in self.results.items():
            if performance_metric in result['metrics'] and size_metric in result['metrics']:
                models.append(model_name)
                performance.append(result['metrics'][performance_metric])
                sizes.append(result['metrics'][size_metric] / 1e6)  # Convert to millions
        
        if not models:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(sizes, performance, s=100, alpha=0.6)
        
        for i, model in enumerate(models):
            ax.annotate(model, (sizes[i], performance[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Model Size (M parameters)', fontsize=12)
        ax.set_ylabel(performance_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title('Performance vs Model Size', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / 'plots' / 'performance_vs_size.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved plot: {plot_path}")
    
    def save_report(self):
        """Save comprehensive benchmark report."""
        report = f"# Benchmark Report: {self.benchmark_name}\n\n"
        report += f"**Created:** {self.metadata['created_at']}\n\n"
        report += "## Model Comparison\n\n"
        report += self.create_comparison_table()
        report += "\n## Summary\n\n"
        report += f"Total models compared: {len(self.results)}\n\n"
        
        # Find best model for each metric
        all_metrics = set()
        for result in self.results.values():
            all_metrics.update(result['metrics'].keys())
        
        report += "### Best Models by Metric\n\n"
        for metric in sorted(all_metrics):
            best_model = None
            best_value = None
            for model_name, result in self.results.items():
                if metric in result['metrics']:
                    value = result['metrics'][metric]
                    if best_value is None or value > best_value:
                        best_value = value
                        best_model = model_name
            if best_model:
                report += f"- **{metric}**: {best_model} ({best_value:.4f})\n"
        
        # Save report
        with open(self.output_dir / 'REPORT.md', 'w') as f:
            f.write(report)
        
        print(f"📄 Saved benchmark report: {self.output_dir / 'REPORT.md'}")
