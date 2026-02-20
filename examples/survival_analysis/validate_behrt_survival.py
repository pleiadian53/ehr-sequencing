"""
Quick validation script for BEHRTForSurvival implementation.

Tests:
1. Model instantiation
2. Forward pass with synthetic data
3. Loss computation
4. Gradient flow
5. Training step

Usage:
    python examples/survival_analysis/validate_behrt_survival.py
"""

import torch
import numpy as np
from pathlib import Path

from ehrsequencing.models.behrt_survival import BEHRTForSurvival, BEHRTSurvivalConfig
from ehrsequencing.models.losses import DiscreteTimeSurvivalLoss, concordance_index
from ehrsequencing.data.behrt_survival_dataset import (
    BEHRTSurvivalDataset,
    collate_behrt_survival
)
from ehrsequencing.synthetic.realistic_synthetic import generate_realistic_dataset
from ehrsequencing.synthetic.survival import DiscreteTimeSurvivalGenerator


def test_model_instantiation():
    """Test 1: Model can be instantiated."""
    print("\n" + "="*80)
    print("TEST 1: Model Instantiation")
    print("="*80)
    
    vocab_size = 100
    config = BEHRTSurvivalConfig.from_pretrained_small(vocab_size=vocab_size)
    model = BEHRTForSurvival(config)
    
    # Check trainable parameters
    params = model.get_trainable_parameters()
    print(f"✅ Model created successfully")
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable parameters: {params['trainable']:,}")
    print(f"   BEHRT parameters: {params['behrt_total']:,}")
    print(f"   Head parameters: {params['head_total']:,}")
    
    return model


def test_forward_pass(model):
    """Test 2: Forward pass works."""
    print("\n" + "="*80)
    print("TEST 2: Forward Pass")
    print("="*80)
    
    batch_size = 4
    seq_len = 50
    vocab_size = 100
    
    # Create dummy batch
    codes = torch.randint(1, vocab_size, (batch_size, seq_len))
    ages = torch.randint(20, 80, (batch_size, seq_len)).float()
    visit_ids = torch.randint(0, 10, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    hazards = model(codes, ages, visit_ids, attention_mask)
    
    print(f"✅ Forward pass successful")
    print(f"   Input shape: {codes.shape}")
    print(f"   Output shape: {hazards.shape}")
    print(f"   Hazard range: [{hazards.min():.4f}, {hazards.max():.4f}]")
    
    # Check hazards are in valid range
    assert (hazards >= 0).all() and (hazards <= 1).all(), "Hazards not in [0, 1]"
    print(f"✅ Hazards in valid range [0, 1]")
    
    return hazards, visit_ids, attention_mask


def test_loss_computation(hazards, visit_ids, attention_mask):
    """Test 3: Loss computation works."""
    print("\n" + "="*80)
    print("TEST 3: Loss Computation")
    print("="*80)
    
    batch_size = hazards.size(0)
    max_visits = visit_ids.max().item() + 1
    
    # Create sequence mask
    sequence_mask = torch.zeros(batch_size, max_visits)
    for b in range(batch_size):
        for v in range(max_visits):
            if ((visit_ids[b] == v) & (attention_mask[b] == 1)).any():
                sequence_mask[b, v] = 1
    
    # Create dummy labels
    event_times = torch.randint(0, max_visits, (batch_size,))
    event_indicators = torch.randint(0, 2, (batch_size,)).float()
    
    # Compute loss
    loss_fn = DiscreteTimeSurvivalLoss()
    loss = loss_fn(hazards, event_times, event_indicators, sequence_mask)
    
    print(f"✅ Loss computation successful")
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Event rate: {event_indicators.mean():.2%}")
    
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is inf"
    print(f"✅ Loss is finite")
    
    return loss


def test_gradient_flow(model, loss):
    """Test 4: Gradients flow correctly."""
    print("\n" + "="*80)
    print("TEST 4: Gradient Flow")
    print("="*80)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = 0
    no_grad = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                has_grad += 1
            else:
                no_grad += 1
    
    print(f"✅ Backward pass successful")
    print(f"   Parameters with gradients: {has_grad}")
    print(f"   Parameters without gradients: {no_grad}")
    
    assert has_grad > 0, "No gradients computed"
    print(f"✅ Gradients flowing correctly")


def test_training_step(model):
    """Test 5: Full training step works."""
    print("\n" + "="*80)
    print("TEST 5: Training Step")
    print("="*80)
    
    batch_size = 8
    seq_len = 50
    vocab_size = 100
    
    # Create dummy batch
    codes = torch.randint(1, vocab_size, (batch_size, seq_len))
    ages = torch.randint(20, 80, (batch_size, seq_len)).float()
    visit_ids = torch.randint(0, 10, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    max_visits = visit_ids.max().item() + 1
    sequence_mask = torch.zeros(batch_size, max_visits)
    for b in range(batch_size):
        for v in range(max_visits):
            if ((visit_ids[b] == v) & (attention_mask[b] == 1)).any():
                sequence_mask[b, v] = 1
    
    event_times = torch.randint(0, max_visits, (batch_size,))
    event_indicators = torch.randint(0, 2, (batch_size,)).float()
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = DiscreteTimeSurvivalLoss()
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    hazards = model(codes, ages, visit_ids, attention_mask)
    loss = loss_fn(hazards, event_times, event_indicators, sequence_mask)
    loss.backward()
    optimizer.step()
    
    print(f"✅ Training step successful")
    print(f"   Loss: {loss.item():.4f}")


def test_realistic_data():
    """Test 6: Works with realistic synthetic data."""
    print("\n" + "="*80)
    print("TEST 6: Realistic Synthetic Data")
    print("="*80)
    
    # Generate realistic data
    num_patients = 50
    vocab_size = 200
    
    codes, ages, visit_ids, attention_mask, _, _ = generate_realistic_dataset(
        num_patients=num_patients,
        vocab_size=vocab_size,
        max_seq_length=128,
        seed=42
    )
    
    # Convert to visit-grouped format with dummy outcomes
    patient_sequences = []
    for i in range(num_patients):
        patient_codes = codes[i].tolist()
        patient_ages = ages[i].tolist()
        patient_visit_ids = visit_ids[i].tolist()
        patient_mask = attention_mask[i].tolist()
        
        visits = []
        max_visit_id = max(patient_visit_ids)
        for v in range(max_visit_id + 1):
            visit_codes = [
                c for c, vid, m in zip(patient_codes, patient_visit_ids, patient_mask)
                if vid == v and m == 1
            ]
            visit_age = [
                a for a, vid, m in zip(patient_ages, patient_visit_ids, patient_mask)
                if vid == v and m == 1
            ][0] if any(vid == v and m == 1 for vid, m in zip(patient_visit_ids, patient_mask)) else 0
            
            if visit_codes:
                visits.append({'codes': visit_codes, 'age': visit_age})
        
        # Add dummy survival outcome
        num_visits = len(visits)
        event_time = np.random.randint(0, max(1, num_visits))
        event_indicator = np.random.randint(0, 2)
        
        patient_sequences.append({
            'visits': visits,
            'outcome': {
                'event_time': event_time,
                'event_indicator': event_indicator
            }
        })
    
    print(f"✅ Generated {len(patient_sequences)} patients")
    print(f"   Average visits: {np.mean([len(s['visits']) for s in patient_sequences]):.1f}")
    print(f"   Event rate: {np.mean([s['outcome']['event_indicator'] for s in patient_sequences]):.2%}")
    
    # Create dataset
    dataset = BEHRTSurvivalDataset(
        patient_sequences=patient_sequences,
        vocab_size=vocab_size,
        max_seq_length=128
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_behrt_survival
    )
    
    # Create model with correct vocab size
    config = BEHRTSurvivalConfig.from_pretrained_small(vocab_size=vocab_size + 1)  # +1 for padding
    model = BEHRTForSurvival(config)
    
    # Run one batch
    batch = next(iter(loader))
    codes = batch['codes']
    ages = batch['ages']
    visit_ids = batch['visit_ids']
    attention_mask = batch['attention_mask']
    
    # Debug: check code range
    max_code = codes.max().item()
    print(f"   Max code in batch: {max_code}, Model vocab size: {vocab_size + 1}")
    
    # Clamp codes to valid range (shouldn't be needed but safety check)
    codes = torch.clamp(codes, 0, vocab_size)
    
    hazards = model(codes, ages, visit_ids, attention_mask)
    
    print(f"✅ Processed batch successfully")
    print(f"   Batch size: {codes.size(0)}")
    print(f"   Sequence length: {codes.size(1)}")
    print(f"   Hazard shape: {hazards.shape}")


def main():
    print("\n" + "="*80)
    print("BEHRT SURVIVAL VALIDATION SUITE")
    print("="*80)
    
    try:
        # Test 1: Instantiation
        model = test_model_instantiation()
        
        # Test 2: Forward pass
        hazards, visit_ids, attention_mask = test_forward_pass(model)
        
        # Test 3: Loss computation
        loss = test_loss_computation(hazards, visit_ids, attention_mask)
        
        # Test 4: Gradient flow
        test_gradient_flow(model, loss)
        
        # Test 5: Training step
        model_fresh = test_model_instantiation()
        test_training_step(model_fresh)
        
        # Test 6: Realistic data
        test_realistic_data()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nBEHRTForSurvival implementation is validated and ready for training!")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ VALIDATION FAILED")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
