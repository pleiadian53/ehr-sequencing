"""
Visualization utilities for benchmarking EHR models.

This module provides plotting functions for comparing training runs,
performance metrics, and model behaviors.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class BenchmarkVisualizer:
    """
    Create visualizations for benchmark comparisons.
    
    This class provides methods for plotting training curves, performance metrics,
    ROC curves, PR curves, and other comparison visualizations.
    
    Example:
        >>> visualizer = BenchmarkVisualizer(output_dir='experiments/plots')
        >>> visualizer.plot_training_curves(tracker.get_all_runs())
        >>> visualizer.plot_performance_metrics(tracker.get_all_runs())
    """
    
    def __init__(self, output_dir: str = "experiments/plots", style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        try:
            plt.style.use(style)
        except:
            pass  # Use default if style not available
        
        sns.set_palette("husl")
    
    def plot_training_curves(self, runs: Dict[str, Dict], filename: str = 'training_curves.png'):
        """
        Plot training curves for all runs.
        
        Args:
            runs: Dictionary mapping run names to their data
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curves
        for name, run in runs.items():
            axes[0, 0].plot(run['train_losses'], label=f"{name} (train)", linewidth=2)
            axes[0, 1].plot(run['val_losses'], label=f"{name} (val)", linewidth=2)
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Validation Loss Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy curves
        for name, run in runs.items():
            axes[1, 0].plot(run['train_accs'], label=f"{name} (train)", linewidth=2)
            axes[1, 1].plot(run['val_accs'], label=f"{name} (val)", linewidth=2)
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Training Accuracy Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Validation Accuracy Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved training curves: {save_path}")
    
    def plot_performance_metrics(
        self,
        runs: Dict[str, Dict],
        metrics: Optional[List[str]] = None,
        filename: str = 'performance_metrics.png'
    ):
        """
        Plot performance metrics comparison.
        
        Args:
            runs: Dictionary mapping run names to their data
            metrics: List of metrics to plot (default: ['roc_auc', 'pr_auc', 'average_precision'])
            filename: Output filename
        """
        if metrics is None:
            metrics = ['roc_auc', 'pr_auc', 'average_precision']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(metrics))
        width = 0.8 / len(runs)
        
        runs_list = list(runs.items())
        
        for i, (name, run) in enumerate(runs_list):
            values = [run['final_metrics'].get(m, 0) for m in metrics]
            offset = width * (i - len(runs_list)/2 + 0.5)
            ax.bar(x + offset, values, width, label=name)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved performance metrics: {save_path}")
    
    def plot_roc_curves(
        self,
        roc_data: Dict[str, Dict[str, np.ndarray]],
        filename: str = 'roc_curves.png'
    ):
        """
        Plot ROC curves for all runs.
        
        Args:
            roc_data: Dictionary mapping run names to ROC data
                     Each entry should have 'fpr', 'tpr', 'auc' keys
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name, data in roc_data.items():
            fpr, tpr, auc_score = data['fpr'], data['tpr'], data['auc']
            ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_score:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved ROC curves: {save_path}")
    
    def plot_pr_curves(
        self,
        pr_data: Dict[str, Dict[str, np.ndarray]],
        filename: str = 'pr_curves.png'
    ):
        """
        Plot Precision-Recall curves for all runs.
        
        Args:
            pr_data: Dictionary mapping run names to PR data
                    Each entry should have 'precision', 'recall', 'auc' keys
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name, data in pr_data.items():
            precision, recall, auc_score = data['precision'], data['recall'], data['auc']
            ax.plot(recall, precision, linewidth=2, label=f"{name} (AUC = {auc_score:.3f})")
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves Comparison')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved PR curves: {save_path}")
    
    def plot_convergence_comparison(
        self,
        runs: Dict[str, Dict],
        metric: str = 'val_losses',
        filename: str = 'convergence.png'
    ):
        """
        Plot convergence comparison for a specific metric.
        
        Args:
            runs: Dictionary mapping run names to their data
            metric: Metric to plot ('val_losses', 'val_accs', etc.)
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for name, run in runs.items():
            if metric in run and run[metric]:
                ax.plot(run[metric], label=name, linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Convergence Comparison: {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved convergence plot: {save_path}")
    
    def plot_training_time_comparison(
        self,
        runs: Dict[str, Dict],
        filename: str = 'training_time.png'
    ):
        """
        Plot training time comparison.
        
        Args:
            runs: Dictionary mapping run names to their data
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(runs.keys())
        times = [runs[name]['training_time'] / 60 for name in names]  # Convert to minutes
        
        bars = ax.bar(names, times, color='steelblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.1f}m',
                   ha='center', va='bottom')
        
        ax.set_ylabel('Training Time (minutes)')
        ax.set_title('Training Time Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"⏱️  Saved training time plot: {save_path}")
    
    def plot_all(
        self,
        runs: Dict[str, Dict],
        roc_data: Optional[Dict] = None,
        pr_data: Optional[Dict] = None
    ):
        """
        Generate all standard plots.
        
        Args:
            runs: Dictionary mapping run names to their data
            roc_data: Optional ROC curve data
            pr_data: Optional PR curve data
        """
        print("\n📊 Generating all benchmark visualizations...")
        
        self.plot_training_curves(runs)
        self.plot_performance_metrics(runs)
        self.plot_convergence_comparison(runs, metric='val_losses', 
                                        filename='convergence_loss.png')
        self.plot_convergence_comparison(runs, metric='val_accs',
                                        filename='convergence_accuracy.png')
        self.plot_training_time_comparison(runs)
        
        if roc_data:
            self.plot_roc_curves(roc_data)
        
        if pr_data:
            self.plot_pr_curves(pr_data)
        
        print(f"✅ All visualizations saved to: {self.output_dir}")
