"""
Unified metrics for benchmarking EHR models across different frameworks.

This module provides consistent metric computation for comparing models from
different libraries (ehrsequencing, PyHealth, etc.).
"""

from typing import Dict, List, Optional
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class UnifiedMetrics:
    """
    Compute unified metrics for EHR model benchmarking.
    
    This class ensures that metrics are computed consistently across different
    model implementations, enabling fair comparison.
    
    Example:
        >>> metrics = UnifiedMetrics()
        >>> results = metrics.compute_mlm_metrics(predictions, labels, attention_mask)
        >>> print(results['accuracy'])
    """
    
    @staticmethod
    def compute_mlm_metrics(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vocab_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute metrics for Masked Language Modeling task.
        
        Args:
            predictions: Model predictions [batch, seq_len] or [batch, seq_len, vocab_size]
            labels: Ground truth labels [batch, seq_len]
            attention_mask: Optional mask [batch, seq_len]
            vocab_size: Vocabulary size (required if predictions are logits)
        
        Returns:
            Dictionary containing:
                - accuracy: Standard accuracy
                - top_5_accuracy: Top-5 accuracy
                - macro_f1: Macro-averaged F1
                - weighted_f1: Weighted F1
                - perplexity: Perplexity (if logits provided)
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.cpu().numpy()
        
        # Handle logits vs predictions
        if len(predictions.shape) == 3:  # [batch, seq, vocab]
            logits = predictions
            predictions = np.argmax(logits, axis=-1)
        else:
            logits = None
        
        # Create mask for valid positions (not padding, not -100)
        if attention_mask is not None:
            mask = (labels != -100) & (attention_mask == 1)
        else:
            mask = (labels != -100)
        
        # Flatten arrays
        pred_flat = predictions[mask]
        label_flat = labels[mask]
        
        if len(pred_flat) == 0:
            return {
                'accuracy': 0.0,
                'top_5_accuracy': 0.0,
                'macro_f1': 0.0,
                'weighted_f1': 0.0,
                'perplexity': float('inf')
            }
        
        # Compute metrics
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(label_flat, pred_flat)
        
        # Top-5 accuracy (if logits available)
        if logits is not None:
            logits_flat = logits.reshape(-1, logits.shape[-1])[mask.flatten()]
            top5_preds = np.argsort(logits_flat, axis=-1)[:, -5:]
            metrics['top_5_accuracy'] = np.mean([
                label in preds for label, preds in zip(label_flat, top5_preds)
            ])
        else:
            metrics['top_5_accuracy'] = None
        
        # F1 scores (with zero_division to handle missing classes)
        try:
            metrics['macro_f1'] = f1_score(
                label_flat, pred_flat, average='macro', zero_division=0
            )
            metrics['weighted_f1'] = f1_score(
                label_flat, pred_flat, average='weighted', zero_division=0
            )
        except Exception:
            metrics['macro_f1'] = 0.0
            metrics['weighted_f1'] = 0.0
        
        # Perplexity (if logits available)
        if logits is not None:
            logits_flat = logits.reshape(-1, logits.shape[-1])[mask.flatten()]
            log_probs = torch.nn.functional.log_softmax(
                torch.from_numpy(logits_flat), dim=-1
            )
            nll = torch.nn.functional.nll_loss(
                log_probs, torch.from_numpy(label_flat), reduction='mean'
            )
            metrics['perplexity'] = torch.exp(nll).item()
        else:
            metrics['perplexity'] = None
        
        return metrics
    
    @staticmethod
    def compute_classification_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        average: str = 'binary'
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Predicted labels
            labels: Ground truth labels
            average: Averaging strategy ('binary', 'macro', 'weighted')
        
        Returns:
            Dictionary of classification metrics
        """
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average=average, zero_division=0),
            'recall': recall_score(labels, predictions, average=average, zero_division=0),
            'f1': f1_score(labels, predictions, average=average, zero_division=0)
        }
    
    @staticmethod
    def compare_models(
        results: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare results from multiple models.
        
        Args:
            results: Dictionary mapping model names to their metrics
            metrics: List of metrics to compare (None = all)
        
        Returns:
            Dictionary with comparison statistics
        """
        if metrics is None:
            # Get all metrics from first model
            metrics = list(next(iter(results.values())).keys())
        
        comparison = {}
        
        for metric in metrics:
            values = {
                model: results[model].get(metric, None)
                for model in results
                if results[model].get(metric) is not None
            }
            
            if not values:
                continue
            
            # Find best and worst
            best_model = max(values, key=values.get)
            worst_model = min(values, key=values.get)
            
            comparison[metric] = {
                'best_model': best_model,
                'best_value': values[best_model],
                'worst_model': worst_model,
                'worst_value': values[worst_model],
                'improvement': values[best_model] - values[worst_model],
                'relative_improvement': (
                    (values[best_model] - values[worst_model]) / values[worst_model] * 100
                    if values[worst_model] != 0 else float('inf')
                ),
                'all_values': values
            }
        
        return comparison
