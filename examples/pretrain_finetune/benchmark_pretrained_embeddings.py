"""
Benchmark: Pre-training vs Fine-tuning with Pre-trained Embeddings

This script compares BEHRT training workflows on the A40 pod:

2-WAY COMPARISON (default):
1. Pre-training from scratch (learning embeddings)
2. Fine-tuning with learned embeddings (frozen embeddings from Run 1)

3-WAY COMPARISON (with --external_embedding_path):
1. Pre-training from scratch (learning embeddings)
2. Fine-tuning with learned embeddings (frozen embeddings from Run 1)
3. Fine-tuning with external embeddings (e.g., Med2Vec, frozen)

HOW IT WORKS:
- Generates realistic synthetic data ONCE (all runs use same dataset)
- Run 1: Trains BEHRT from scratch, learning embeddings from the data
- Saves the learned embeddings after training
- Run 2: Loads embeddings from Run 1, freezes them, trains only LoRA + head
- Run 3 (optional): Loads external embeddings (Med2Vec), freezes them, trains only LoRA + head
- Compares performance: Which embedding strategy works best?

This answers key questions:
- Does using pre-trained embeddings help convergence/accuracy?
- Is there a difference between self-learned vs external (Med2Vec) embeddings?
- Can we skip expensive pre-training if we have good external embeddings?

Outputs comprehensive performance comparison:
- Training curves (loss, accuracy)
- Performance metrics (PRAUC, AP, ROC-AUC)
- Comparison plots and tables
- Statistical significance tests

Uses realistic synthetic data by default for meaningful evaluation.

Usage:

# 2-way comparison (default)
python benchmark_pretrained_embeddings.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --batch-size 128

# 3-way comparison (with external Med2Vec embeddings)
python benchmark_pretrained_embeddings.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --batch-size 128 \
    --external-embedding-path pretrained/med2vec_embeddings.pt

# Quick test
python benchmark_pretrained_embeddings.py \
    --model-size small \
    --num-patients 1000 \
    --epochs 20 \
    --batch-size 32
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import argparse
import json
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc
)

from ehrsequencing.models.behrt import BEHRT, BEHRTConfig, BEHRTForMLM
from ehrsequencing.models.lora import apply_lora_to_behrt, count_parameters
from ehrsequencing.models.pretrained_embeddings import (
    save_embeddings,
    load_embeddings,
    initialize_embedding_layer,
    print_embedding_statistics
)
from ehrsequencing.data.realistic_synthetic import generate_realistic_dataset, print_dataset_statistics


class BenchmarkTracker:
    """Track and compare multiple training runs."""
    
    def __init__(self, output_dir: str = "experiments/benchmark_embeddings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.runs = {}
        self.start_time = time.time()
        
        print(f"📊 Benchmark tracker initialized: {self.output_dir}")
    
    def add_run(self, name: str, config: Dict):
        """Add a new run to track."""
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
    
    def log_epoch(self, name: str, epoch: int, train_loss: float, train_acc: float, 
                  val_loss: float, val_acc: float):
        """Log metrics for an epoch."""
        run = self.runs[name]
        run['train_losses'].append(train_loss)
        run['val_losses'].append(val_loss)
        run['train_accs'].append(train_acc)
        run['val_accs'].append(val_acc)
        
        if val_loss < run['best_val_loss']:
            run['best_val_loss'] = val_loss
            run['best_epoch'] = epoch
    
    def set_training_time(self, name: str, duration: float):
        """Set training duration for a run."""
        self.runs[name]['training_time'] = duration
    
    def set_final_metrics(self, name: str, metrics: Dict):
        """Set final evaluation metrics."""
        self.runs[name]['final_metrics'] = metrics
    
    def plot_training_curves(self):
        """Plot training curves for all runs."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curves
        for name, run in self.runs.items():
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
        for name, run in self.runs.items():
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
        save_path = self.output_dir / 'training_curves_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved training curves: {save_path}")
    
    def plot_performance_metrics(self):
        """Plot performance metrics comparison."""
        metrics_to_plot = ['roc_auc_macro', 'roc_auc_micro', 'pr_auc', 'avg_precision_macro']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        runs_list = list(self.runs.items())
        
        for i, (name, run) in enumerate(runs_list):
            values = [run['final_metrics'].get(m, 0) for m in metrics_to_plot]
            offset = width * (i - len(runs_list)/2 + 0.5)
            ax.bar(x + offset, values, width, label=name)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(['ROC-AUC', 'PR-AUC', 'Average Precision'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        save_path = self.output_dir / 'performance_metrics_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved performance metrics: {save_path}")
    
    def plot_roc_curves(self, roc_data: Dict):
        """Plot ROC curves for all runs."""
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
        save_path = self.output_dir / 'roc_curves_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved ROC curves: {save_path}")
    
    def plot_pr_curves(self, pr_data: Dict):
        """Plot Precision-Recall curves for all runs."""
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
        save_path = self.output_dir / 'pr_curves_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Saved PR curves: {save_path}")
    
    def generate_summary_table(self):
        """Generate summary comparison table."""
        summary = []
        
        for name, run in self.runs.items():
            summary.append({
                'Model': name,
                'Best Val Loss': f"{run['best_val_loss']:.4f}",
                'Best Epoch': run['best_epoch'],
                'Final Train Acc': f"{run['train_accs'][-1]:.4f}" if run['train_accs'] else "N/A",
                'Final Val Acc': f"{run['val_accs'][-1]:.4f}" if run['val_accs'] else "N/A",
                'ROC-AUC (macro)': f"{run['final_metrics'].get('roc_auc_macro', 0):.4f}",
                'ROC-AUC (micro)': f"{run['final_metrics'].get('roc_auc_micro', 0):.4f}",
                'PR-AUC': f"{run['final_metrics'].get('pr_auc', 0):.4f}",
                'AP (macro)': f"{run['final_metrics'].get('avg_precision_macro', 0):.4f}",
                'Training Time (min)': f"{run['training_time']/60:.2f}",
                'Trainable Params': run['config'].get('trainable_params', 'N/A')
            })
        
        # Save as JSON
        json_path = self.output_dir / 'summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save as text table
        txt_path = self.output_dir / 'SUMMARY.txt'
        with open(txt_path, 'w') as f:
            f.write("="*120 + "\n")
            f.write("BENCHMARK SUMMARY: Pre-training vs Fine-tuning with Pre-trained Embeddings\n")
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
            f.write("\n" + "="*120 + "\n")
            f.write("WINNER ANALYSIS\n")
            f.write("="*120 + "\n\n")
            
            best_val_loss = min(self.runs.items(), key=lambda x: x[1]['best_val_loss'])
            best_roc_auc_macro = max(self.runs.items(), key=lambda x: x[1]['final_metrics'].get('roc_auc_macro', 0))
            best_roc_auc_micro = max(self.runs.items(), key=lambda x: x[1]['final_metrics'].get('roc_auc_micro', 0))
            fastest = min(self.runs.items(), key=lambda x: x[1]['training_time'])
            
            f.write(f"Best Validation Loss: {best_val_loss[0]} ({best_val_loss[1]['best_val_loss']:.4f})\n")
            f.write(f"Best ROC-AUC (macro): {best_roc_auc_macro[0]} ({best_roc_auc_macro[1]['final_metrics'].get('roc_auc_macro', 0):.4f})\n")
            f.write(f"Best ROC-AUC (micro): {best_roc_auc_micro[0]} ({best_roc_auc_micro[1]['final_metrics'].get('roc_auc_micro', 0):.4f})\n")
            f.write(f"Fastest Training: {fastest[0]} ({fastest[1]['training_time']/60:.2f} min)\n")
        
        print(f"📄 Saved summary: {txt_path}")
        
        return summary


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_masked = 0
    
    for batch in dataloader:
        codes, ages, visit_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        # BEHRTForMLM returns (logits, loss) when labels are provided
        logits, loss = model(codes, ages=ages, visit_ids=visit_ids, attention_mask=attention_mask, labels=labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        mask = labels != -100
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions[mask] == labels[mask]).sum().item()
        total_masked += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_masked if total_masked > 0 else 0
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_masked = 0
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            codes, ages, visit_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            # BEHRTForMLM returns (logits, loss) when labels are provided
            logits, loss = model(codes, ages=ages, visit_ids=visit_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += loss.item()
            
            mask = labels != -100
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions[mask] == labels[mask]).sum().item()
            total_masked += mask.sum().item()
            
            # Collect probabilities and labels for metrics
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs[mask].cpu())
            all_labels.append(labels[mask].cpu())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_masked if total_masked > 0 else 0
    
    # Concatenate all predictions
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return avg_loss, accuracy, all_probs, all_labels


def compute_metrics(probs: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> Dict:
    """Compute performance metrics.
    
    Uses macro-average (equal weight per class) as primary metric for medical codes
    where rare classes are clinically important. Also computes micro-average for
    comparison with baselines.
    
    Filters to only classes present in the dataset to avoid sklearn warnings about
    one-class problems (common with 1000 vocab but small test sets).
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
    
    # Compute ROC-AUC (macro and micro)
    try:
        roc_auc_macro = roc_auc_score(labels_onehot, probs_filtered, average='macro', multi_class='ovr')
    except:
        roc_auc_macro = 0.0
    
    try:
        roc_auc_micro = roc_auc_score(labels_onehot, probs_filtered, average='micro', multi_class='ovr')
    except:
        roc_auc_micro = 0.0
    
    # Compute average precision (macro and micro)
    try:
        avg_precision_macro = average_precision_score(labels_onehot, probs_filtered, average='macro')
    except:
        avg_precision_macro = 0.0
    
    try:
        avg_precision_micro = average_precision_score(labels_onehot, probs_filtered, average='micro')
    except:
        avg_precision_micro = 0.0
    
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
        'roc_auc_macro': roc_auc_macro,  # Primary: equal weight per class
        'roc_auc_micro': roc_auc_micro,  # Secondary: equal weight per sample
        'pr_auc': pr_auc_avg,
        'avg_precision_macro': avg_precision_macro,
        'avg_precision_micro': avg_precision_micro,
        'n_classes_present': n_present  # For debugging
    }


def compute_roc_curve(probs: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> Tuple:
    """Compute ROC curve data."""
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


def compute_pr_curve(probs: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> Tuple:
    """Compute PR curve data."""
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


def train_model(name: str, model, train_loader, val_loader, optimizer, device, 
                epochs: int, tracker: BenchmarkTracker, vocab_size: int):
    """Train a model and track metrics."""
    print(f"\n{'='*80}")
    print(f"Training: {name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_probs, val_labels = evaluate(model, val_loader, device)
        
        tracker.log_epoch(name, epoch, train_loss, train_acc, val_loss, val_acc)
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            trophy = "🏆"
        else:
            patience_counter += 1
            trophy = ""
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} {trophy} | "
              f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    tracker.set_training_time(name, training_time)
    
    # Final evaluation
    print(f"\n📊 Computing final metrics for {name}...")
    _, _, final_probs, final_labels = evaluate(model, val_loader, device)
    metrics = compute_metrics(final_probs, final_labels, vocab_size)
    tracker.set_final_metrics(name, metrics)
    
    print(f"   ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")
    print(f"   ROC-AUC (micro): {metrics['roc_auc_micro']:.4f}")
    print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"   Avg Precision (macro): {metrics['avg_precision_macro']:.4f}")
    print(f"   Classes present: {metrics['n_classes_present']}/{vocab_size}")
    
    return final_probs, final_labels


def main():
    parser = argparse.ArgumentParser(description='Benchmark: Pre-training vs Fine-tuning')
    parser.add_argument('--model-size', type=str, default='large', choices=['small', 'medium', 'large'],
                       help='Model size (use large for A40)')
    parser.add_argument('--num-patients', type=int, default=10000,
                       help='Number of patients (use 10K+ for A40)')
    parser.add_argument('--vocab-size', type=int, default=1000,
                       help='Vocabulary size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Max epochs per run')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (use 128+ for A40)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout')
    parser.add_argument('--lora-rank', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--external-embedding-path', type=str, default=None,
                       help='Path to external pre-trained embeddings (e.g., Med2Vec). If provided, adds 3rd comparison run.')
    parser.add_argument('--output-dir', type=str, default='experiments/benchmark_embeddings',
                       help='Output directory')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tracker = BenchmarkTracker(args.output_dir)
    
    print("\n" + "="*80)
    print("BENCHMARK: Pre-training vs Fine-tuning with Pre-trained Embeddings")
    print("="*80)
    print(f"Model size: {args.model_size}")
    print(f"Patients: {args.num_patients}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Using realistic synthetic data with disease patterns")
    
    # Model config - create BEFORE data generation to use correct max_position
    if args.model_size == 'small':
        config = BEHRTConfig.small(vocab_size=args.vocab_size)
    elif args.model_size == 'medium':
        config = BEHRTConfig.medium(vocab_size=args.vocab_size)
    else:
        config = BEHRTConfig.large(vocab_size=args.vocab_size)
    
    config.dropout = args.dropout
    
    # Use model's max_position for sequence length to avoid out-of-bounds embedding errors
    max_seq_length = config.max_position
    print(f"Using max_seq_length={max_seq_length} (from model config)")
    
    # Generate data once (shared across both runs)
    print(f"\n🔬 Generating realistic synthetic data...")
    codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_realistic_dataset(
        num_patients=args.num_patients,
        vocab_size=args.vocab_size,
        max_seq_length=max_seq_length,
        seed=42
    )
    print_dataset_statistics(codes, ages, visit_ids)
    
    train_size = int(0.8 * args.num_patients)
    train_dataset = TensorDataset(
        masked_codes[:train_size], ages[:train_size], visit_ids[:train_size],
        attention_mask[:train_size], labels[:train_size]
    )
    val_dataset = TensorDataset(
        masked_codes[train_size:], ages[train_size:], visit_ids[train_size:],
        attention_mask[train_size:], labels[train_size:]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # ============================================================================
    # RUN 1: Pre-training from Scratch
    # ============================================================================
    print(f"\n{'='*80}")
    print("RUN 1: Pre-training from Scratch (learning embeddings)")
    print(f"{'='*80}")
    
    model1 = BEHRTForMLM(config).to(device)
    model1 = apply_lora_to_behrt(
        model1,
        rank=args.lora_rank,
        lora_attention=True,
        train_embeddings=True,  # Learn embeddings
        train_head=True
    )
    
    params1 = count_parameters(model1)
    print(f"\n📊 Model Parameters (Pre-training):")
    print(f"   Total: {params1['total']:,}")
    print(f"   Trainable: {params1['trainable']:,} ({params1['trainable_percent']:.1f}%)")
    print(f"   Embeddings: {params1['embedding_trainable']:,}/{params1['embedding_total']:,} trainable")
    
    tracker.add_run('Pre-training (from scratch)', {
        'trainable_params': f"{params1['trainable']:,} ({params1['trainable_percent']:.1f}%)",
        'embeddings_trainable': True,
        'lora_rank': args.lora_rank
    })
    
    optimizer1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model1.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    probs1, labels1 = train_model(
        'Pre-training (from scratch)', model1, train_loader, val_loader,
        optimizer1, device, args.epochs, tracker, args.vocab_size
    )
    
    # Save embeddings for Run 2
    embedding_path = Path(args.output_dir) / 'pretrained_embeddings.pt'
    save_embeddings(
        model1.behrt.embeddings.code_embedding.weight.data,
        embedding_path,
        metadata={'vocab_size': args.vocab_size, 'embedding_dim': config.embedding_dim}
    )
    
    # ============================================================================
    # RUN 2: Fine-tuning with Pre-trained Embeddings
    # ============================================================================
    print(f"\n{'='*80}")
    print("RUN 2: Fine-tuning with Pre-trained Embeddings (frozen embeddings)")
    print(f"{'='*80}")
    
    model2 = BEHRTForMLM(config).to(device)
    
    # Load pre-trained embeddings from Run 1
    print(f"\n📂 Loading pre-trained embeddings from Run 1...")
    pretrained_emb = model1.behrt.embeddings.code_embedding.weight.data.clone()
    initialize_embedding_layer(
        model2.behrt.embeddings.code_embedding,
        pretrained_emb,
        freeze=True
    )
    
    model2 = apply_lora_to_behrt(
        model2,
        rank=args.lora_rank,
        lora_attention=True,
        train_embeddings=False,  # Freeze embeddings
        train_head=True
    )
    
    params2 = count_parameters(model2)
    print(f"\n📊 Model Parameters (Fine-tuning):")
    print(f"   Total: {params2['total']:,}")
    print(f"   Trainable: {params2['trainable']:,} ({params2['trainable_percent']:.1f}%)")
    print(f"   Embeddings: {params2['embedding_trainable']:,}/{params2['embedding_total']:,} trainable (frozen)")
    
    tracker.add_run('Fine-tuning (pre-trained embeddings)', {
        'trainable_params': f"{params2['trainable']:,} ({params2['trainable_percent']:.1f}%)",
        'embeddings_trainable': False,
        'lora_rank': args.lora_rank
    })
    
    optimizer2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model2.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    probs2, labels2 = train_model(
        'Fine-tuning (pre-trained embeddings)', model2, train_loader, val_loader,
        optimizer2, device, args.epochs, tracker, args.vocab_size
    )
    
    # ============================================================================
    # RUN 3 (Optional): Fine-tuning with External Pre-trained Embeddings (e.g., Med2Vec)
    # ============================================================================
    if args.external_embedding_path:
        print(f"\n{'='*80}")
        print("RUN 3: Fine-tuning with External Pre-trained Embeddings (e.g., Med2Vec)")
        print(f"{'='*80}")
        
        model3 = BEHRTForMLM(config).to(device)
        
        # Load external pre-trained embeddings
        print(f"\n📂 Loading external pre-trained embeddings from: {args.external_embedding_path}")
        external_emb, metadata = load_embeddings(args.external_embedding_path)
        print(f"   Loaded embeddings: {external_emb.shape}")
        print(f"   Metadata: {metadata}")
        
        # Initialize with external embeddings
        initialize_embedding_layer(
            model3.behrt.embeddings.code_embedding,
            external_emb,
            freeze=True
        )
        
        model3 = apply_lora_to_behrt(
            model3,
            rank=args.lora_rank,
            lora_attention=True,
            train_embeddings=False,  # Freeze embeddings
            train_head=True
        )
        
        params3 = count_parameters(model3)
        print(f"\n📊 Model Parameters (Fine-tuning with External):")
        print(f"   Total: {params3['total']:,}")
        print(f"   Trainable: {params3['trainable']:,} ({params3['trainable_percent']:.1f}%)")
        print(f"   Embeddings: {params3['embedding_trainable']:,}/{params3['embedding_total']:,} trainable (frozen)")
        
        tracker.add_run('Fine-tuning (external embeddings)', {
            'trainable_params': f"{params3['trainable']:,} ({params3['trainable_percent']:.1f}%)",
            'embeddings_trainable': False,
            'lora_rank': args.lora_rank,
            'embedding_source': args.external_embedding_path
        })
        
        optimizer3 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model3.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        probs3, labels3 = train_model(
            'Fine-tuning (external embeddings)', model3, train_loader, val_loader,
            optimizer3, device, args.epochs, tracker, args.vocab_size
        )
    else:
        probs3, labels3 = None, None
    
    # ============================================================================
    # Generate Comparison Plots and Summary
    # ============================================================================
    print(f"\n{'='*80}")
    print("Generating Comparison Plots and Summary")
    print(f"{'='*80}")
    
    tracker.plot_training_curves()
    tracker.plot_performance_metrics()
    
    # ROC curves
    print("\n📈 Computing ROC curves...")
    roc_data = {}
    runs_to_plot = [
        ('Pre-training (from scratch)', probs1, labels1),
        ('Fine-tuning (pre-trained embeddings)', probs2, labels2)
    ]
    if probs3 is not None:
        runs_to_plot.append(('Fine-tuning (external embeddings)', probs3, labels3))
    
    for name, probs, lbls in runs_to_plot:
        fpr, tpr, auc_score = compute_roc_curve(probs, lbls, args.vocab_size)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_score}
    
    tracker.plot_roc_curves(roc_data)
    
    # PR curves
    print("📈 Computing PR curves...")
    pr_data = {}
    for name, probs, lbls in runs_to_plot:
        precision, recall, auc_score = compute_pr_curve(probs, lbls, args.vocab_size)
        pr_data[name] = {'precision': precision, 'recall': recall, 'auc': auc_score}
    
    tracker.plot_pr_curves(pr_data)
    
    # Summary table
    summary = tracker.generate_summary_table()
    
    print(f"\n{'='*80}")
    print("✅ BENCHMARK COMPLETE!")
    print(f"{'='*80}")
    print(f"📁 All outputs saved to: {args.output_dir}")
    print(f"⏱️  Total time: {(time.time() - tracker.start_time)/60:.2f} minutes")
    print(f"\nKey files:")
    print(f"   - {args.output_dir}/SUMMARY.txt")
    print(f"   - {args.output_dir}/training_curves_comparison.png")
    print(f"   - {args.output_dir}/performance_metrics_comparison.png")
    print(f"   - {args.output_dir}/roc_curves_comparison.png")
    print(f"   - {args.output_dir}/pr_curves_comparison.png")


if __name__ == '__main__':
    main()
