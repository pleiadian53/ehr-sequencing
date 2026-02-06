"""
Benchmark script comparing loss functions for BEHRT survival analysis.

Compares three approaches:
1. NLL (standard) - Optimizes calibration
2. Pairwise Ranking - Directly optimizes C-index
3. Hybrid (NLL + Ranking) - Best of both worlds

This script answers the research question:
"Can we directly optimize for higher C-index using pairwise ranking loss?"

Usage:
    python benchmark_loss_functions.py --num-patients 5000 --epochs 100
"""

import argparse
import os
import json
import time
from pathlib import Path
from typing import Dict, List
import subprocess

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark loss functions for survival analysis')
    
    # Data
    parser.add_argument('--num-patients', type=int, default=5000,
                        help='Number of synthetic patients')
    parser.add_argument('--vocab-size', type=int, default=1000,
                        help='Vocabulary size')
    
    # Model
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'medium', 'large'],
                        help='BEHRT model size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    # Ranking loss hyperparameters
    parser.add_argument('--margins', type=float, nargs='+', default=[0.05, 0.1, 0.2],
                        help='Margins to test for ranking loss')
    parser.add_argument('--lambda-ranks', type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.5],
                        help='Lambda values to test for hybrid loss')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='experiments/loss_comparison',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def run_experiment(
    loss_type: str,
    args,
    margin: float = None,
    lambda_rank: float = None
) -> Dict:
    """Run a single training experiment."""
    
    # Build command
    cmd = [
        'python', 'examples/survival_analysis/train_behrt_survival.py',
        '--loss', loss_type,
        '--num-patients', str(args.num_patients),
        '--vocab-size', str(args.vocab_size),
        '--model-size', args.model_size,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--seed', str(args.seed),
        '--output-dir', args.output_dir
    ]
    
    # Add loss-specific parameters
    if loss_type == 'ranking' and margin is not None:
        cmd.extend(['--margin', str(margin)])
    elif loss_type == 'hybrid':
        if margin is not None:
            cmd.extend(['--margin', str(margin)])
        if lambda_rank is not None:
            cmd.extend(['--lambda-rank', str(lambda_rank)])
    
    # Run experiment
    print(f"\n{'='*80}")
    print(f"Running: {loss_type}", end='')
    if margin is not None:
        print(f" (margin={margin})", end='')
    if lambda_rank is not None:
        print(f" (lambda_rank={lambda_rank})", end='')
    print(f"\n{'='*80}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Error running experiment: {result.stderr}")
        return None
    
    # Load results
    exp_name = f"{loss_type}_{args.model_size}"
    results_path = Path(args.output_dir) / exp_name / 'results.json'
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        results['elapsed_time'] = elapsed
        results['loss_type'] = loss_type
        results['margin'] = margin
        results['lambda_rank'] = lambda_rank
        return results
    else:
        print(f"Results file not found: {results_path}")
        return None


def plot_comparison(all_results: List[Dict], output_dir: Path):
    """Create comparison plots."""
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'Loss Type': r['loss_type'],
            'Margin': r.get('margin', 'N/A'),
            'Lambda Rank': r.get('lambda_rank', 'N/A'),
            'Test C-index': r['test_c_index'],
            'Test Loss': r['test_loss'],
            'Training Time (s)': r['training_time'],
            'Best Val C-index': r['best_val_c_index']
        }
        for r in all_results if r is not None
    ])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Loss Function Comparison for BEHRT Survival Analysis', fontsize=16, y=1.02)
    
    # 1. Test C-index comparison
    ax = axes[0, 0]
    df_plot = df.copy()
    df_plot['Label'] = df_plot.apply(
        lambda x: f"{x['Loss Type']}\n(λ={x['Lambda Rank']})" if x['Lambda Rank'] != 'N/A'
        else f"{x['Loss Type']}\n(m={x['Margin']})" if x['Margin'] != 'N/A'
        else x['Loss Type'],
        axis=1
    )
    sns.barplot(data=df_plot, x='Label', y='Test C-index', ax=ax, palette='Set2')
    ax.set_title('Test C-index (Higher is Better)', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('C-index')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Training time comparison
    ax = axes[0, 1]
    sns.barplot(data=df_plot, x='Label', y='Training Time (s)', ax=ax, palette='Set2')
    ax.set_title('Training Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Time (seconds)')
    ax.tick_params(axis='x', rotation=45)
    
    # 3. C-index vs Training Time scatter
    ax = axes[1, 0]
    for loss_type in df['Loss Type'].unique():
        df_loss = df[df['Loss Type'] == loss_type]
        ax.scatter(df_loss['Training Time (s)'], df_loss['Test C-index'],
                  label=loss_type, s=100, alpha=0.7)
    ax.set_xlabel('Training Time (s)')
    ax.set_ylabel('Test C-index')
    ax.set_title('Efficiency vs Performance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary statistics
    summary = df.groupby('Loss Type').agg({
        'Test C-index': ['mean', 'std', 'max'],
        'Training Time (s)': 'mean'
    }).round(4)
    
    table_data = []
    table_data.append(['Loss Type', 'Mean C-index', 'Std C-index', 'Max C-index', 'Avg Time (s)'])
    for loss_type in summary.index:
        row = [
            loss_type,
            f"{summary.loc[loss_type, ('Test C-index', 'mean')]:.4f}",
            f"{summary.loc[loss_type, ('Test C-index', 'std')]:.4f}",
            f"{summary.loc[loss_type, ('Test C-index', 'max')]:.4f}",
            f"{summary.loc[loss_type, ('Training Time (s)', 'mean')]:.1f}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_dir / 'loss_comparison.png'}")
    
    return df


def generate_report(df: pd.DataFrame, output_dir: Path):
    """Generate markdown report with findings."""
    
    report = f"""# Loss Function Comparison for BEHRT Survival Analysis

## Experiment Setup

- **Model**: BEHRT (small)
- **Dataset**: Synthetic EHR with survival outcomes
- **Objective**: Compare NLL, Pairwise Ranking, and Hybrid losses

## Research Question

**Can we directly optimize for higher C-index using pairwise ranking loss?**

## Results Summary

### Overall Performance

| Loss Type | Mean C-index | Std C-index | Max C-index | Avg Training Time (s) |
|-----------|--------------|-------------|-------------|-----------------------|
"""
    
    summary = df.groupby('Loss Type').agg({
        'Test C-index': ['mean', 'std', 'max'],
        'Training Time (s)': 'mean'
    }).round(4)
    
    for loss_type in summary.index:
        mean_ci = summary.loc[loss_type, ('Test C-index', 'mean')]
        std_ci = summary.loc[loss_type, ('Test C-index', 'std')]
        max_ci = summary.loc[loss_type, ('Test C-index', 'max')]
        avg_time = summary.loc[loss_type, ('Training Time (s)', 'mean')]
        report += f"| {loss_type} | {mean_ci:.4f} | {std_ci:.4f} | {max_ci:.4f} | {avg_time:.1f} |\n"
    
    # Find best configuration
    best_idx = df['Test C-index'].idxmax()
    best_config = df.iloc[best_idx]
    
    report += f"""

### Best Configuration

- **Loss Type**: {best_config['Loss Type']}
- **Test C-index**: {best_config['Test C-index']:.4f}
- **Training Time**: {best_config['Training Time (s)']:.1f}s
"""
    
    if best_config['Lambda Rank'] != 'N/A':
        report += f"- **Lambda Rank**: {best_config['Lambda Rank']}\n"
    if best_config['Margin'] != 'N/A':
        report += f"- **Margin**: {best_config['Margin']}\n"
    
    report += """

## Key Findings

### 1. Direct C-index Optimization

"""
    
    # Compare NLL vs Ranking
    nll_cindex = df[df['Loss Type'] == 'nll']['Test C-index'].mean()
    rank_cindex = df[df['Loss Type'] == 'ranking']['Test C-index'].mean()
    hybrid_cindex = df[df['Loss Type'] == 'hybrid']['Test C-index'].mean()
    
    if rank_cindex > nll_cindex:
        report += f"✅ **Pairwise ranking loss achieves higher C-index** ({rank_cindex:.4f}) than NLL ({nll_cindex:.4f})\n\n"
        report += "This confirms that directly optimizing for ranking (C-index) can improve discrimination.\n\n"
    else:
        report += f"❌ **Pairwise ranking loss does not outperform NLL**: Ranking={rank_cindex:.4f}, NLL={nll_cindex:.4f}\n\n"
        report += "This suggests that calibration (NLL) may indirectly improve discrimination, or that the ranking loss needs better tuning.\n\n"
    
    report += """
### 2. Hybrid Approach

"""
    
    if hybrid_cindex > max(nll_cindex, rank_cindex):
        report += f"✅ **Hybrid loss achieves best performance** ({hybrid_cindex:.4f})\n\n"
        report += "Combining NLL and ranking losses provides the best of both worlds: calibration and discrimination.\n\n"
    else:
        report += f"**Hybrid loss**: {hybrid_cindex:.4f}\n\n"
        report += "Hybrid approach does not outperform single-objective losses in this experiment.\n\n"
    
    report += """
### 3. Training Efficiency

"""
    
    nll_time = df[df['Loss Type'] == 'nll']['Training Time (s)'].mean()
    rank_time = df[df['Loss Type'] == 'ranking']['Training Time (s)'].mean()
    hybrid_time = df[df['Loss Type'] == 'hybrid']['Training Time (s)'].mean()
    
    report += f"- **NLL**: {nll_time:.1f}s\n"
    report += f"- **Ranking**: {rank_time:.1f}s\n"
    report += f"- **Hybrid**: {hybrid_time:.1f}s\n\n"
    
    if rank_time > nll_time * 1.2:
        report += "⚠️ Ranking loss is significantly slower due to O(n²) pairwise comparisons.\n\n"
    
    report += """
## Recommendations

"""
    
    if best_config['Loss Type'] == 'nll':
        report += """
1. **Use NLL loss as default** - It provides good C-index while being efficient and well-calibrated.
2. Consider ranking loss only if C-index is critical and calibration is less important.
3. Monitor both calibration (Brier score) and discrimination (C-index) during training.
"""
    elif best_config['Loss Type'] == 'ranking':
        report += """
1. **Pairwise ranking loss can improve C-index** - Use when discrimination is the primary objective.
2. Be aware of potential calibration issues - probabilities may be less meaningful.
3. Tune margin parameter carefully (tested: {}).
""".format(', '.join(map(str, df[df['Loss Type'] == 'ranking']['Margin'].unique())))
    else:
        report += f"""
1. **Hybrid loss provides best performance** - Use λ_rank={best_config['Lambda Rank']:.2f} as starting point.
2. Balance calibration and discrimination by tuning λ_rank.
3. Start with small λ_rank (0.01-0.1) and increase if C-index plateaus.
"""
    
    report += """

## Conclusion

"""
    
    if rank_cindex > nll_cindex:
        report += """
**Yes, we can directly optimize for higher C-index using pairwise ranking loss.**

The experiments confirm that ranking-based losses can improve discrimination (C-index) compared to standard NLL. However, the best approach depends on the specific use case:

- **Clinical risk prediction**: Use hybrid loss for balanced calibration and discrimination
- **Patient stratification**: Use ranking loss for maximum discrimination
- **General survival analysis**: Use NLL for simplicity and interpretability

The key insight is that NLL and C-index measure different properties (calibration vs discrimination), and the choice of loss function should align with the downstream task requirements.
"""
    else:
        report += """
**The results are mixed - ranking loss does not clearly outperform NLL in this experiment.**

This could be due to:
1. Dataset characteristics (synthetic data may not reflect real-world complexity)
2. Hyperparameter tuning (margin and λ_rank may need optimization)
3. Model capacity (small BEHRT may benefit more from calibration)

Further experiments with real EHR data and larger models are recommended.
"""
    
    # Save report
    with open(output_dir / 'REPORT.md', 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to {output_dir / 'REPORT.md'}")


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    all_results = []
    
    # 1. Run NLL baseline
    print("\n" + "="*80)
    print("EXPERIMENT 1: NLL Loss (Baseline)")
    print("="*80)
    result = run_experiment('nll', args)
    if result:
        all_results.append(result)
    
    # 2. Run Pairwise Ranking with different margins
    print("\n" + "="*80)
    print("EXPERIMENT 2: Pairwise Ranking Loss")
    print("="*80)
    for margin in args.margins:
        result = run_experiment('ranking', args, margin=margin)
        if result:
            all_results.append(result)
    
    # 3. Run Hybrid with different lambda_rank values
    print("\n" + "="*80)
    print("EXPERIMENT 3: Hybrid Loss (NLL + Ranking)")
    print("="*80)
    for lambda_rank in args.lambda_ranks:
        result = run_experiment('hybrid', args, margin=0.1, lambda_rank=lambda_rank)
        if result:
            all_results.append(result)
    
    # Save all results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate plots and report
    if all_results:
        df = plot_comparison(all_results, output_dir)
        generate_report(df, output_dir)
        
        print("\n" + "="*80)
        print("BENCHMARK COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {output_dir}")
        print(f"- Plots: {output_dir / 'loss_comparison.png'}")
        print(f"- Report: {output_dir / 'REPORT.md'}")
        print(f"- Data: {output_dir / 'all_results.json'}")
    else:
        print("\n❌ No results to analyze")


if __name__ == '__main__':
    main()
