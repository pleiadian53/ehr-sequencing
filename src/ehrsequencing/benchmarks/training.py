"""
Training and evaluation utilities for benchmarking EHR models.

This module provides reusable training loops and evaluation functions
for benchmarking different model configurations.
"""

from typing import Dict, Tuple, Optional, Callable
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc
)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Optional[Callable] = None
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        loss_fn: Optional custom loss function (default: CrossEntropyLoss)
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_correct = 0
    total_masked = 0
    
    for batch in dataloader:
        codes, ages, visit_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        
        # Handle both model types: (logits, loss) or just outputs
        model_output = model(codes, ages=ages, visit_ids=visit_ids, attention_mask=attention_mask, labels=labels)
        
        if isinstance(model_output, tuple) and len(model_output) == 2:
            # BEHRTForMLM returns (logits, loss)
            outputs, loss = model_output
        else:
            # Standard model returns just outputs
            outputs = model_output
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Compute accuracy
        mask = labels != -100
        predictions = outputs.argmax(dim=-1)
        total_correct += (predictions[mask] == labels[mask]).sum().item()
        total_masked += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_masked if total_masked > 0 else 0
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Optional[Callable] = None,
    return_predictions: bool = True
) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Evaluate on validation/test set.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to evaluate on
        loss_fn: Optional custom loss function
        return_predictions: Whether to return predictions and labels
    
    Returns:
        Tuple of (loss, accuracy, probabilities, labels)
        If return_predictions=False, probabilities and labels are None
    """
    model.eval()
    
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_correct = 0
    total_masked = 0
    
    all_probs = [] if return_predictions else None
    all_labels = [] if return_predictions else None
    
    with torch.no_grad():
        for batch in dataloader:
            codes, ages, visit_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            # Handle both model types: (logits, loss) or just outputs
            model_output = model(codes, ages=ages, visit_ids=visit_ids, attention_mask=attention_mask, labels=labels)
            
            if isinstance(model_output, tuple) and len(model_output) == 2:
                # BEHRTForMLM returns (logits, loss)
                outputs, loss = model_output
            else:
                # Standard model returns just outputs
                outputs = model_output
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            
            mask = labels != -100
            predictions = outputs.argmax(dim=-1)
            total_correct += (predictions[mask] == labels[mask]).sum().item()
            total_masked += mask.sum().item()
            
            if return_predictions:
                # Collect probabilities and labels for metrics
                probs = torch.softmax(outputs, dim=-1)
                all_probs.append(probs[mask].cpu())
                all_labels.append(labels[mask].cpu())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_masked if total_masked > 0 else 0
    
    if return_predictions:
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
    
    return avg_loss, accuracy, all_probs, all_labels


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    tracker: Optional[object] = None,
    vocab_size: Optional[int] = None,
    patience: int = 10,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Train a model with early stopping and metric tracking.
    
    Args:
        name: Name for this training run
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        epochs: Maximum number of epochs
        tracker: Optional BenchmarkTracker instance
        vocab_size: Vocabulary size (for computing metrics)
        patience: Early stopping patience
        verbose: Whether to print progress
    
    Returns:
        Tuple of (final_probabilities, final_labels) from validation set
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Training: {name}")
        print(f"{'='*80}")
    
    start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_probs, val_labels = evaluate(
            model, val_loader, device, return_predictions=True
        )
        
        # Track metrics
        if tracker is not None:
            tracker.log_epoch(name, epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Early stopping check
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            trophy = "🏆"
        else:
            patience_counter += 1
            trophy = ""
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} {trophy} | "
                  f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    if tracker is not None:
        tracker.set_training_time(name, training_time)
    
    # Final evaluation
    if verbose:
        print(f"\n📊 Computing final metrics for {name}...")
    
    _, _, final_probs, final_labels = evaluate(
        model, val_loader, device, return_predictions=True
    )
    
    if vocab_size is not None and tracker is not None:
        metrics = compute_metrics(final_probs, final_labels, vocab_size)
        tracker.set_final_metrics(name, metrics)
        
        if verbose:
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
            print(f"   Average Precision: {metrics['average_precision']:.4f}")
    
    return final_probs, final_labels


def compute_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int
) -> Dict[str, float]:
    """
    Compute performance metrics for multi-class classification.
    
    Filters to only classes present in the dataset to avoid sklearn warnings about
    one-class problems (common with large vocab but small test sets).
    
    Args:
        probs: Predicted probabilities [N, vocab_size]
        labels: Ground truth labels [N]
        vocab_size: Vocabulary size
    
    Returns:
        Dictionary containing roc_auc, pr_auc, and average_precision
    """
    # Convert to numpy
    probs_np = probs.numpy()
    labels_np = labels.numpy()
    
    # Get unique classes present in labels (avoids one-class warnings)
    present_classes = np.unique(labels_np)
    n_present = len(present_classes)
    
    # Create one-hot encoding for present classes only
    labels_onehot = np.zeros((len(labels_np), n_present))
    for i, cls in enumerate(present_classes):
        labels_onehot[labels_np == cls, i] = 1
    
    # Filter probabilities to present classes
    probs_filtered = probs_np[:, present_classes]
    
    # Compute metrics
    try:
        roc_auc = roc_auc_score(labels_onehot, probs_filtered, average='macro', multi_class='ovr')
    except:
        roc_auc = 0.0
    
    try:
        avg_precision = average_precision_score(labels_onehot, probs_filtered, average='macro')
    except:
        avg_precision = 0.0
    
    # For PR-AUC, compute per-class and average (already filtered to present classes)
    pr_aucs = []
    for i in range(n_present):
        if labels_onehot[:, i].sum() > 0:  # Double-check (should always be true now)
            precision, recall, _ = precision_recall_curve(labels_onehot[:, i], probs_filtered[:, i])
            pr_auc = auc(recall, precision)
            if not np.isnan(pr_auc):
                pr_aucs.append(pr_auc)
    
    pr_auc_avg = np.mean(pr_aucs) if pr_aucs else 0.0
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc_avg,
        'average_precision': avg_precision
    }


def compute_roc_curve(
    probs: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute macro-averaged ROC curve.
    
    Args:
        probs: Predicted probabilities [N, vocab_size]
        labels: Ground truth labels [N]
        vocab_size: Vocabulary size
    
    Returns:
        Tuple of (fpr, tpr, auc_score)
    """
    probs_np = probs.numpy()
    labels_np = labels.numpy()
    
    labels_onehot = np.zeros((len(labels_np), vocab_size))
    labels_onehot[np.arange(len(labels_np)), labels_np] = 1
    
    # Compute macro-average ROC curve
    all_fpr = []
    all_tpr = []
    
    for i in range(min(vocab_size, probs_np.shape[1])):
        if labels_onehot[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(labels_onehot[:, i], probs_np[:, i])
            all_fpr.append(fpr)
            all_tpr.append(tpr)
    
    # Interpolate all ROC curves at common FPR points
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []
    
    for fpr, tpr in zip(all_fpr, all_tpr):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
    
    mean_tpr = np.mean(interp_tprs, axis=0) if interp_tprs else np.zeros_like(mean_fpr)
    mean_tpr[-1] = 1.0
    
    auc_score = auc(mean_fpr, mean_tpr)
    
    return mean_fpr, mean_tpr, auc_score


def compute_pr_curve(
    probs: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute macro-averaged Precision-Recall curve.
    
    Args:
        probs: Predicted probabilities [N, vocab_size]
        labels: Ground truth labels [N]
        vocab_size: Vocabulary size
    
    Returns:
        Tuple of (precision, recall, auc_score)
    """
    probs_np = probs.numpy()
    labels_np = labels.numpy()
    
    labels_onehot = np.zeros((len(labels_np), vocab_size))
    labels_onehot[np.arange(len(labels_np)), labels_np] = 1
    
    all_precision = []
    all_recall = []
    
    for i in range(min(vocab_size, probs_np.shape[1])):
        if labels_onehot[:, i].sum() > 0:
            precision, recall, _ = precision_recall_curve(labels_onehot[:, i], probs_np[:, i])
            all_precision.append(precision)
            all_recall.append(recall)
    
    # Interpolate
    mean_recall = np.linspace(0, 1, 100)
    interp_precisions = []
    
    for precision, recall in zip(all_precision, all_recall):
        # Reverse for interpolation
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        interp_precisions.append(interp_precision)
    
    mean_precision = np.mean(interp_precisions, axis=0) if interp_precisions else np.zeros_like(mean_recall)
    
    auc_score = auc(mean_recall, mean_precision)
    
    return mean_precision, mean_recall, auc_score
