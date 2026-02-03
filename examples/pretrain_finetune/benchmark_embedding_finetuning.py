"""
Benchmark: Embedding Fine-tuning Strategy Comparison

This script compares different strategies for using pre-trained embeddings:

3-WAY COMPARISON:
1. Train from scratch (baseline - learn embeddings from data)
2. Load pre-trained embeddings, FREEZE them (reduced capacity)
3. Load pre-trained embeddings, FINE-TUNE them (transfer learning)

HOW IT WORKS:
- Generates realistic synthetic data ONCE (all runs use same dataset)
- Run 1: Trains BEHRT from scratch, learning embeddings from the data
- Saves the learned embeddings after training
- Run 2: Loads embeddings from Run 1, FREEZES them, trains only LoRA + head
- Run 3: Loads embeddings from Run 1, FINE-TUNES them, trains full model
- Compares performance: Which embedding strategy works best?

This answers key questions:
- Does freezing embeddings hurt performance? (Yes, expected)
- Does fine-tuning pre-trained embeddings match or beat training from scratch?
- What's the trade-off between frozen (faster) vs fine-tuned (better)?

EXPECTED RESULTS:
- Fine-tuned ≥ Scratch > Frozen (performance ranking)
- Fine-tuned should converge faster than scratch (fewer epochs)
- Frozen should show degraded performance (fewer trainable params)

Outputs comprehensive performance comparison:
- Training curves (loss, accuracy)
- Performance metrics (PRAUC, AP, ROC-AUC)
- Comparison plots and tables

Uses realistic synthetic data by default for meaningful evaluation.

Usage:

# Full 3-way comparison (recommended)
python benchmark_embedding_finetuning.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --batch-size 128

# Quick test
python benchmark_embedding_finetuning.py \
    --model-size small \
    --num-patients 1000 \
    --epochs 20 \
    --batch-size 32

Note: For testing transfer learning across different datasets, use benchmark_transfer_learning.py
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
from ehrsequencing.benchmarks import (
    BenchmarkTracker,
    BenchmarkVisualizer,
    train_epoch,
    evaluate,
    compute_metrics,
    compute_roc_curve,
    compute_pr_curve
)


class CustomBenchmarkVisualizer:
    """
    Custom visualization methods for embedding comparison benchmark.
    
    Extends the base BenchmarkTracker with specialized plots for this experiment.
    These methods will eventually be moved to BenchmarkVisualizer in the shared module.
    """
    
    def __init__(self, tracker: BenchmarkTracker):
        """Initialize with a BenchmarkTracker instance."""
        self.tracker = tracker
        self.output_dir = tracker.output_dir
        self.runs = tracker.runs
    
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


# train_epoch, evaluate, compute_metrics, compute_roc_curve, compute_pr_curve
# are now imported from ehrsequencing.benchmarks.training


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
    _, _, final_probs, final_labels = evaluate(model, val_loader, device, return_predictions=True)
    metrics = compute_metrics(final_probs, final_labels, vocab_size)
    tracker.set_final_metrics(name, metrics)
    
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"   Avg Precision: {metrics['average_precision']:.4f}")
    
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
    # RUN 3: Fine-tuning with Pre-trained Embeddings (FINE-TUNED, not frozen)
    # ============================================================================
    print(f"\n{'='*80}")
    print("RUN 3: Fine-tuning with Pre-trained Embeddings (fine-tuned embeddings)")
    print(f"{'='*80}")
    
    model3 = BEHRTForMLM(config).to(device)
    
    # Load pre-trained embeddings from Run 1
    print(f"\n📂 Loading pre-trained embeddings from Run 1 (will be fine-tuned)...")
    pretrained_emb_finetune = model1.behrt.embeddings.code_embedding.weight.data.clone()
    initialize_embedding_layer(
        model3.behrt.embeddings.code_embedding,
        pretrained_emb_finetune,
        freeze=False  # Allow fine-tuning
    )
    
    model3 = apply_lora_to_behrt(
        model3,
        rank=args.lora_rank,
        lora_attention=True,
        train_embeddings=True,  # Fine-tune embeddings
        train_head=True
    )
    
    params3 = count_parameters(model3)
    print(f"\n📊 Model Parameters (Fine-tuning with trainable embeddings):")
    print(f"   Total: {params3['total']:,}")
    print(f"   Trainable: {params3['trainable']:,} ({params3['trainable_percent']:.1f}%)")
    print(f"   Embeddings: {params3['embedding_trainable']:,}/{params3['embedding_total']:,} trainable (fine-tuned)")
    
    tracker.add_run('Fine-tuning (fine-tuned embeddings)', {
        'trainable_params': f"{params3['trainable']:,} ({params3['trainable_percent']:.1f}%)",
        'embeddings_trainable': True,
        'lora_rank': args.lora_rank
    })
    
    optimizer3 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model3.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    probs3, labels3 = train_model(
        'Fine-tuning (fine-tuned embeddings)', model3, train_loader, val_loader,
        optimizer3, device, args.epochs, tracker, args.vocab_size
    )
    
    # ============================================================================
    # RUN 4 (Optional): Fine-tuning with External Pre-trained Embeddings (e.g., Med2Vec)
    # ============================================================================
    if args.external_embedding_path:
        print(f"\n{'='*80}")
        print("RUN 4: Fine-tuning with External Pre-trained Embeddings (e.g., Med2Vec)")
        print(f"{'='*80}")
        
        model4 = BEHRTForMLM(config).to(device)
        
        # Load external pre-trained embeddings
        print(f"\n📂 Loading external pre-trained embeddings from: {args.external_embedding_path}")
        external_emb, metadata = load_embeddings(args.external_embedding_path)
        print(f"   Loaded embeddings: {external_emb.shape}")
        print(f"   Metadata: {metadata}")
        
        # Initialize with external embeddings
        initialize_embedding_layer(
            model4.behrt.embeddings.code_embedding,
            external_emb,
            freeze=True
        )
        
        model4 = apply_lora_to_behrt(
            model4,
            rank=args.lora_rank,
            lora_attention=True,
            train_embeddings=False,  # Freeze embeddings
            train_head=True
        )
        
        params4 = count_parameters(model4)
        print(f"\n📊 Model Parameters (Fine-tuning with External):")
        print(f"   Total: {params4['total']:,}")
        print(f"   Trainable: {params4['trainable']:,} ({params4['trainable_percent']:.1f}%)")
        print(f"   Embeddings: {params4['embedding_trainable']:,}/{params4['embedding_total']:,} trainable (frozen)")
        
        tracker.add_run('Fine-tuning (external embeddings)', {
            'trainable_params': f"{params4['trainable']:,} ({params4['trainable_percent']:.1f}%)",
            'embeddings_trainable': False,
            'lora_rank': args.lora_rank,
            'embedding_source': args.external_embedding_path
        })
        
        optimizer4 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model4.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        probs4, labels4 = train_model(
            'Fine-tuning (external embeddings)', model4, train_loader, val_loader,
            optimizer4, device, args.epochs, tracker, args.vocab_size
        )
    else:
        probs4, labels4 = None, None
    
    # ============================================================================
    # Generate Comparison Plots and Summary
    # ============================================================================
    print(f"\n{'='*80}")
    print("Generating Comparison Plots and Summary")
    print(f"{'='*80}")
    
    # Use BenchmarkVisualizer for plotting
    visualizer = BenchmarkVisualizer(output_dir=args.output_dir)
    visualizer.plot_all(tracker.get_all_runs())
    
    # ROC curves
    print("\n📈 Computing ROC curves...")
    roc_data = {}
    runs_to_plot = [
        ('Pre-training (from scratch)', probs1, labels1),
        ('Fine-tuning (pre-trained embeddings)', probs2, labels2),
        ('Fine-tuning (fine-tuned embeddings)', probs3, labels3)
    ]
    if probs4 is not None:
        runs_to_plot.append(('Fine-tuning (external embeddings)', probs4, labels4))
    
    for name, probs, lbls in runs_to_plot:
        fpr, tpr, auc_score = compute_roc_curve(probs, lbls, args.vocab_size)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_score}
    
    # Use CustomBenchmarkVisualizer for custom plots
    custom_viz = CustomBenchmarkVisualizer(tracker)
    custom_viz.plot_roc_curves(roc_data)
    
    # PR curves
    print("📈 Computing PR curves...")
    pr_data = {}
    for name, probs, lbls in runs_to_plot:
        precision, recall, auc_score = compute_pr_curve(probs, lbls, args.vocab_size)
        pr_data[name] = {'precision': precision, 'recall': recall, 'auc': auc_score}
    
    custom_viz.plot_pr_curves(pr_data)
    
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
