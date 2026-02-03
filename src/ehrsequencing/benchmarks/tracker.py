"""
Benchmark tracking and comparison utilities.

This module provides tools for tracking multiple training runs and comparing
their performance across various metrics.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd


class BenchmarkTracker:
    """
    Track and compare multiple training runs.
    
    This class provides a unified interface for tracking metrics across different
    model training runs, enabling easy comparison and visualization.
    
    Example:
        >>> tracker = BenchmarkTracker(output_dir='experiments/comparison')
        >>> tracker.add_run('BEHRT-scratch', config={'model_size': 'large'})
        >>> tracker.log_epoch('BEHRT-scratch', epoch=0, train_loss=2.5, ...)
        >>> tracker.generate_summary_table()
    """
    
    def __init__(self, output_dir: str = "experiments/benchmark"):
        """
        Initialize benchmark tracker.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.runs = {}
        self.start_time = time.time()
        
        print(f"📊 Benchmark tracker initialized: {self.output_dir}")
    
    def add_run(self, name: str, config: Dict[str, Any]):
        """
        Add a new run to track.
        
        Args:
            name: Unique name for this run
            config: Configuration dictionary for this run
        """
        self.runs[name] = {
            'config': config,
            'metrics': [],
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'training_time': 0,
            'final_metrics': {}
        }
        print(f"✅ Added run: {name}")
    
    def log_epoch(
        self,
        name: str,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float
    ):
        """
        Log metrics for an epoch.
        
        Args:
            name: Run name
            epoch: Epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
        """
        if name not in self.runs:
            raise ValueError(f"Run '{name}' not found. Call add_run() first.")
        
        run = self.runs[name]
        run['train_losses'].append(train_loss)
        run['val_losses'].append(val_loss)
        run['train_accs'].append(train_acc)
        run['val_accs'].append(val_acc)
        
        if val_loss < run['best_val_loss']:
            run['best_val_loss'] = val_loss
            run['best_epoch'] = epoch
    
    def set_training_time(self, name: str, duration: float):
        """
        Set training duration for a run.
        
        Args:
            name: Run name
            duration: Training duration in seconds
        """
        if name not in self.runs:
            raise ValueError(f"Run '{name}' not found.")
        
        self.runs[name]['training_time'] = duration
    
    def set_final_metrics(self, name: str, metrics: Dict[str, float]):
        """
        Set final evaluation metrics.
        
        Args:
            name: Run name
            metrics: Dictionary of metric name -> value
        """
        if name not in self.runs:
            raise ValueError(f"Run '{name}' not found.")
        
        self.runs[name]['final_metrics'] = metrics
    
    def get_run(self, name: str) -> Dict[str, Any]:
        """
        Get data for a specific run.
        
        Args:
            name: Run name
        
        Returns:
            Dictionary containing run data
        """
        if name not in self.runs:
            raise ValueError(f"Run '{name}' not found.")
        
        return self.runs[name]
    
    def get_all_runs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get data for all runs.
        
        Returns:
            Dictionary mapping run names to their data
        """
        return self.runs
    
    def generate_summary_table(self) -> List[Dict[str, Any]]:
        """
        Generate summary comparison table.
        
        Returns:
            List of dictionaries containing summary data for each run
        """
        summary = []
        
        for name, run in self.runs.items():
            summary.append({
                'Model': name,
                'Best Val Loss': f"{run['best_val_loss']:.4f}",
                'Best Epoch': run['best_epoch'],
                'Final Train Acc': f"{run['train_accs'][-1]:.4f}" if run['train_accs'] else "N/A",
                'Final Val Acc': f"{run['val_accs'][-1]:.4f}" if run['val_accs'] else "N/A",
                'ROC-AUC': f"{run['final_metrics'].get('roc_auc', 0):.4f}",
                'PR-AUC': f"{run['final_metrics'].get('pr_auc', 0):.4f}",
                'AP': f"{run['final_metrics'].get('average_precision', 0):.4f}",
                'Training Time (min)': f"{run['training_time']/60:.2f}",
                'Trainable Params': run['config'].get('trainable_params', 'N/A')
            })
        
        # Save as JSON
        json_path = self.output_dir / 'summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save as CSV
        csv_path = self.output_dir / 'summary.csv'
        pd.DataFrame(summary).to_csv(csv_path, index=False)
        
        # Save as text table
        self._save_text_summary(summary)
        
        print(f"📄 Saved summary: {self.output_dir / 'SUMMARY.txt'}")
        
        return summary
    
    def _save_text_summary(self, summary: List[Dict[str, Any]]):
        """Save summary as formatted text table."""
        txt_path = self.output_dir / 'SUMMARY.txt'
        
        with open(txt_path, 'w') as f:
            f.write("="*120 + "\n")
            f.write("BENCHMARK SUMMARY\n")
            f.write("="*120 + "\n\n")
            
            # Header
            headers = list(summary[0].keys())
            f.write(" | ".join(f"{h:20s}" for h in headers) + "\n")
            f.write("-" * 120 + "\n")
            
            # Rows
            for row in summary:
                f.write(" | ".join(f"{str(row[h]):20s}" for h in headers) + "\n")
            
            f.write("\n" + "="*120 + "\n")
            f.write(f"Total benchmark time: {(time.time() - self.start_time)/60:.2f} minutes\n")
            
            # Winner analysis
            if len(self.runs) > 1:
                f.write("\n" + "="*120 + "\n")
                f.write("WINNER ANALYSIS\n")
                f.write("="*120 + "\n\n")
                
                best_val_loss = min(self.runs.items(), key=lambda x: x[1]['best_val_loss'])
                best_roc_auc = max(self.runs.items(), 
                                  key=lambda x: x[1]['final_metrics'].get('roc_auc', 0))
                fastest = min(self.runs.items(), key=lambda x: x[1]['training_time'])
                
                f.write(f"Best Validation Loss: {best_val_loss[0]} "
                       f"({best_val_loss[1]['best_val_loss']:.4f})\n")
                f.write(f"Best ROC-AUC: {best_roc_auc[0]} "
                       f"({best_roc_auc[1]['final_metrics'].get('roc_auc', 0):.4f})\n")
                f.write(f"Fastest Training: {fastest[0]} "
                       f"({fastest[1]['training_time']/60:.2f} min)\n")
    
    def save_state(self, filename: str = 'tracker_state.json'):
        """
        Save tracker state to disk.
        
        Args:
            filename: Filename to save state
        """
        state = {
            'runs': self.runs,
            'start_time': self.start_time,
            'elapsed_time': time.time() - self.start_time
        }
        
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print(f"💾 Saved tracker state: {path}")
    
    def load_state(self, filename: str = 'tracker_state.json'):
        """
        Load tracker state from disk.
        
        Args:
            filename: Filename to load state from
        """
        path = self.output_dir / filename
        
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.runs = state['runs']
        self.start_time = state['start_time']
        
        print(f"📂 Loaded tracker state: {path}")
