"""
Benchmark BEHRT against PyHealth's Transformer.

This script compares ehrsequencing's BEHRT implementation against PyHealth's
generic Transformer on the same MLM task and data.

Key Differences:
- BEHRT: EHR-specific (code + age + visit + segment embeddings)
- PyHealth: Generic transformer (code embeddings only)

Expected Result: BEHRT should outperform PyHealth due to EHR-specific design.

Usage:
    # Basic benchmark
    python examples/benchmarking/benchmark_pyhealth.py
    
    # Custom configuration
    python examples/benchmarking/benchmark_pyhealth.py \
        --model-size large \
        --num-patients 5000 \
        --epochs 50 \
        --batch-size 128
    
    # Use realistic data
    python examples/benchmarking/benchmark_pyhealth.py --realistic-data
"""

import sys
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from ehrsequencing.models.behrt import BEHRTForMLM, BEHRTConfig
from ehrsequencing.models.lora import apply_lora_to_behrt
from ehrsequencing.synthetic.demo_data import generate_demo_dataset, print_demo_dataset_statistics
from ehrsequencing.synthetic.realistic_data import generate_realistic_dataset, print_dataset_statistics

# Import benchmarking tools
try:
    from ehrsequencing.benchmarks import PyHealthAdapter, ModelComparator
    BENCHMARKING_AVAILABLE = True
except ImportError as e:
    BENCHMARKING_AVAILABLE = False
    IMPORT_ERROR = str(e)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark BEHRT vs PyHealth Transformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model configuration
    parser.add_argument('--model-size', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='Model size for both BEHRT and PyHealth')
    parser.add_argument('--use-lora', action='store_true', default=True,
                       help='Use LoRA for BEHRT (default: True)')
    parser.add_argument('--lora-rank', type=int, default=16,
                       help='LoRA rank')
    
    # Data configuration
    parser.add_argument('--num-patients', type=int, default=2000,
                       help='Number of synthetic patients')
    parser.add_argument('--vocab-size', type=int, default=1000,
                       help='Vocabulary size')
    parser.add_argument('--realistic-data', action='store_true',
                       help='Use realistic data instead of demo data')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout probability')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str,
                       default='examples/benchmarking/results',
                       help='Output directory for results')
    
    return parser.parse_args()


def create_behrt_model(args, device):
    """Create BEHRT model."""
    # Model size configurations
    size_configs = {
        'small': {'embedding_dim': 128, 'hidden_dim': 256, 'num_layers': 4, 'num_heads': 4},
        'medium': {'embedding_dim': 256, 'hidden_dim': 512, 'num_layers': 6, 'num_heads': 8},
        'large': {'embedding_dim': 256, 'hidden_dim': 512, 'num_layers': 6, 'num_heads': 8}
    }
    
    config_params = size_configs[args.model_size]
    
    config = BEHRTConfig(
        vocab_size=args.vocab_size,
        embedding_dim=config_params['embedding_dim'],
        hidden_dim=config_params['hidden_dim'],
        num_layers=config_params['num_layers'],
        num_heads=config_params['num_heads'],
        dropout=args.dropout,
        max_position=512,
        num_age_bins=100,
        num_visit_bins=50,
        num_segments=2
    )
    
    model = BEHRTForMLM(config).to(device)
    
    if args.use_lora:
        print(f"Applying LoRA (rank={args.lora_rank}) to BEHRT...")
        model = apply_lora_to_behrt(
            model,
            rank=args.lora_rank,
            lora_attention=True,
            train_embeddings=True,
            train_head=True
        )
    
    return model, config


def create_pyhealth_adapter(args, device):
    """Create PyHealth adapter."""
    # Model size configurations (matching BEHRT)
    size_configs = {
        'small': {'embedding_dim': 128, 'hidden_dim': 256, 'num_layers': 4, 'num_heads': 4},
        'medium': {'embedding_dim': 256, 'hidden_dim': 512, 'num_layers': 6, 'num_heads': 8},
        'large': {'embedding_dim': 256, 'hidden_dim': 512, 'num_layers': 6, 'num_heads': 8}
    }
    
    config_params = size_configs[args.model_size]
    
    config = {
        'vocab_size': args.vocab_size,
        'embedding_dim': config_params['embedding_dim'],
        'hidden_dim': config_params['hidden_dim'],
        'num_layers': config_params['num_layers'],
        'num_heads': config_params['num_heads'],
        'dropout': args.dropout
    }
    
    adapter = PyHealthAdapter(config=config, device=device)
    adapter.build_model()
    
    return adapter


def generate_data(args, max_seq_length=512):
    """Generate synthetic data."""
    print(f"\n🔬 Generating synthetic data...")
    
    if args.realistic_data:
        print("Using realistic synthetic data with disease patterns...")
        codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_realistic_dataset(
            num_patients=args.num_patients,
            vocab_size=args.vocab_size,
            max_seq_length=max_seq_length,
            seed=42
        )
        print_dataset_statistics(codes, ages, visit_ids)
    else:
        print("Using HIGH-SIGNAL demo data with very strong patterns...")
        codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_demo_dataset(
            num_patients=args.num_patients,
            vocab_size=args.vocab_size,
            max_seq_length=max_seq_length,
            seed=42
        )
        print_demo_dataset_statistics(codes, ages, visit_ids)
    
    return codes, ages, visit_ids, attention_mask, masked_codes, labels


def create_dataloaders(codes, ages, visit_ids, attention_mask, labels, batch_size):
    """Create train/val/test dataloaders."""
    num_patients = codes.shape[0]
    train_size = int(0.7 * num_patients)
    val_size = int(0.15 * num_patients)
    
    # Split data
    train_dataset = TensorDataset(
        codes[:train_size],
        ages[:train_size],
        visit_ids[:train_size],
        attention_mask[:train_size],
        labels[:train_size]
    )
    
    val_dataset = TensorDataset(
        codes[train_size:train_size+val_size],
        ages[train_size:train_size+val_size],
        visit_ids[train_size:train_size+val_size],
        attention_mask[train_size:train_size+val_size],
        labels[train_size:train_size+val_size]
    )
    
    test_dataset = TensorDataset(
        codes[train_size+val_size:],
        ages[train_size+val_size:],
        visit_ids[train_size+val_size:],
        attention_mask[train_size+val_size:],
        labels[train_size+val_size:]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n📊 Data splits:")
    print(f"   Train: {len(train_dataset)} patients")
    print(f"   Val: {len(val_dataset)} patients")
    print(f"   Test: {len(test_dataset)} patients")
    
    return train_loader, val_loader, test_loader


def main():
    args = parse_args()
    
    # Check if benchmarking is available
    if not BENCHMARKING_AVAILABLE:
        print("❌ Error: Benchmarking tools not available!")
        print(f"   Import error: {IMPORT_ERROR}")
        print("\n💡 To fix this, install PyHealth:")
        print("   Option 1: pip install pyhealth")
        print("   Option 2: mamba env create -f environment-benchmarking.yml")
        sys.exit(1)
    
    print("="*80)
    print("BEHRT vs PyHealth Transformer Benchmark")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Generate data
    codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_data(args)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        masked_codes, ages, visit_ids, attention_mask, labels, args.batch_size
    )
    
    # Create models
    print(f"\n🏗️  Building models ({args.model_size})...")
    
    print("\n1️⃣  BEHRT (EHR-specific with age/visit embeddings)")
    behrt_model, behrt_config = create_behrt_model(args, device)
    
    print("\n2️⃣  PyHealth Transformer (generic, code-only)")
    pyhealth_adapter = create_pyhealth_adapter(args, device)
    
    # Print model info
    behrt_params = sum(p.numel() for p in behrt_model.parameters() if p.requires_grad)
    pyhealth_params = pyhealth_adapter.count_parameters()
    
    print(f"\n📊 Model Comparison:")
    print(f"   BEHRT trainable params: {behrt_params:,}")
    print(f"   PyHealth trainable params: {pyhealth_params:,}")
    
    # Run benchmark
    print(f"\n🚀 Starting benchmark ({args.epochs} epochs)...")
    print("="*80)
    
    # Note: For now, we'll just train PyHealth since BEHRT needs custom training loop
    # In a full implementation, we'd wrap BEHRT in an adapter too
    
    print("\n⚠️  Note: Full benchmark implementation in progress")
    print("Training PyHealth model as demonstration...")
    
    results = pyhealth_adapter.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Evaluate
    test_results = pyhealth_adapter.evaluate(test_loader)
    
    print(f"\n✅ PyHealth Results:")
    print(f"   Test Accuracy: {test_results['test_accuracy']:.4f}")
    print(f"   Test Loss: {test_results['test_loss']:.4f}")
    
    print(f"\n💡 Next Steps:")
    print("   1. Wrap BEHRT in a similar adapter for fair comparison")
    print("   2. Use ModelComparator to run both models")
    print("   3. Generate comprehensive comparison reports")
    
    print(f"\n📁 Results would be saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
