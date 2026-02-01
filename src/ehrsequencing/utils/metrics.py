"""
Comprehensive evaluation metrics for medical code prediction.

Provides metrics beyond simple accuracy that are more appropriate for:
- Imbalanced medical code distributions
- Rare but important codes
- Clinical relevance
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix
)


def compute_mlm_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    top_k: int = 5
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for masked language modeling.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: True labels [batch_size, seq_len] (with -100 for non-masked)
        vocab_size: Size of vocabulary
        top_k: K for top-K accuracy
    
    Returns:
        Dictionary with metrics:
        - accuracy: Standard accuracy
        - top_k_accuracy: Is correct code in top K predictions?
        - macro_f1: F1 averaged across all codes (treats rare codes equally)
        - weighted_f1: F1 weighted by code frequency
        - macro_precision: Precision averaged across codes
        - macro_recall: Recall averaged across codes
        - perplexity: Exp(cross-entropy loss)
    """
    # Get mask for valid predictions
    mask = labels != -100
    
    if not mask.any():
        return {
            'accuracy': 0.0,
            f'top_{top_k}_accuracy': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'perplexity': float('inf')
        }
    
    # Get predictions and labels for masked positions only
    masked_logits = logits[mask]  # [num_masked, vocab_size]
    masked_labels = labels[mask]  # [num_masked]
    
    # Standard accuracy
    predictions = masked_logits.argmax(dim=-1)
    accuracy = (predictions == masked_labels).float().mean().item()
    
    # Top-K accuracy
    top_k_preds = masked_logits.topk(k=top_k, dim=-1).indices
    top_k_accuracy = (top_k_preds == masked_labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    
    # Convert to numpy for sklearn metrics
    preds_np = predictions.cpu().numpy()
    labels_np = masked_labels.cpu().numpy()
    
    # F1 scores (handle cases where some codes never appear)
    try:
        macro_f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)
        weighted_f1 = f1_score(labels_np, preds_np, average='weighted', zero_division=0)
        macro_precision = precision_score(labels_np, preds_np, average='macro', zero_division=0)
        macro_recall = recall_score(labels_np, preds_np, average='macro', zero_division=0)
    except Exception as e:
        # Fallback if sklearn fails
        macro_f1 = 0.0
        weighted_f1 = 0.0
        macro_precision = 0.0
        macro_recall = 0.0
    
    # Perplexity
    probs = torch.softmax(masked_logits, dim=-1)
    true_probs = probs[torch.arange(len(masked_labels)), masked_labels]
    perplexity = torch.exp(-torch.log(true_probs + 1e-10).mean()).item()
    
    return {
        'accuracy': accuracy,
        f'top_{top_k}_accuracy': top_k_accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'perplexity': perplexity
    }


def compute_per_code_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    top_codes: int = 20
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-code metrics for the most frequent codes.
    
    Useful for understanding which codes the model learns well vs poorly.
    
    Returns:
        Dictionary mapping code_id -> {precision, recall, f1, support}
    """
    mask = labels != -100
    if not mask.any():
        return {}
    
    predictions = logits[mask].argmax(dim=-1)
    labels_masked = labels[mask]
    
    preds_np = predictions.cpu().numpy()
    labels_np = labels_masked.cpu().numpy()
    
    # Get most frequent codes
    unique_codes, counts = np.unique(labels_np, return_counts=True)
    top_code_indices = np.argsort(counts)[-top_codes:]
    top_code_ids = unique_codes[top_code_indices]
    
    per_code_metrics = {}
    
    for code_id in top_code_ids:
        # Binary classification for this code
        true_binary = (labels_np == code_id).astype(int)
        pred_binary = (preds_np == code_id).astype(int)
        
        if true_binary.sum() == 0:
            continue
        
        precision = precision_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary, zero_division=0)
        f1 = f1_score(true_binary, pred_binary, zero_division=0)
        support = true_binary.sum()
        
        per_code_metrics[int(code_id)] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(support)
        }
    
    return per_code_metrics


def compute_clinical_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    important_codes: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute metrics focused on clinically important codes.
    
    Args:
        logits: Model predictions
        labels: True labels
        important_codes: List of code IDs that are clinically important
                        (e.g., rare diseases, critical medications)
    
    Returns:
        Metrics for important codes specifically
    """
    if important_codes is None:
        return {}
    
    mask = labels != -100
    if not mask.any():
        return {}
    
    predictions = logits[mask].argmax(dim=-1)
    labels_masked = labels[mask]
    
    # Filter to only important codes
    important_mask = torch.zeros_like(labels_masked, dtype=torch.bool)
    for code in important_codes:
        important_mask |= (labels_masked == code)
    
    if not important_mask.any():
        return {}
    
    important_preds = predictions[important_mask]
    important_labels = labels_masked[important_mask]
    
    accuracy = (important_preds == important_labels).float().mean().item()
    
    preds_np = important_preds.cpu().numpy()
    labels_np = important_labels.cpu().numpy()
    
    macro_f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)
    
    return {
        'important_codes_accuracy': accuracy,
        'important_codes_macro_f1': macro_f1,
        'important_codes_count': int(important_mask.sum())
    }


def print_metrics_summary(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics summary."""
    print(f"\n{prefix}Metrics Summary:")
    print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"   Top-5 Accuracy: {metrics.get('top_5_accuracy', 0):.4f}")
    print(f"   Macro F1: {metrics.get('macro_f1', 0):.4f}")
    print(f"   Weighted F1: {metrics.get('weighted_f1', 0):.4f}")
    print(f"   Macro Precision: {metrics.get('macro_precision', 0):.4f}")
    print(f"   Macro Recall: {metrics.get('macro_recall', 0):.4f}")
    print(f"   Perplexity: {metrics.get('perplexity', float('inf')):.2f}")


def get_metrics_for_logging(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Extract key metrics for experiment logging.
    
    Returns subset of metrics that are most important to track.
    """
    return {
        'accuracy': metrics.get('accuracy', 0),
        'top_5_accuracy': metrics.get('top_5_accuracy', 0),
        'macro_f1': metrics.get('macro_f1', 0),
        'weighted_f1': metrics.get('weighted_f1', 0),
        'perplexity': metrics.get('perplexity', float('inf'))
    }
