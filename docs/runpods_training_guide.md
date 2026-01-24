# RunPods Training Guide for EHR Survival Models

This guide explains how to train large-scale survival models on cloud GPUs when local resources are insufficient.

## When to Use Cloud Training

### Local System Limitations

**Symptoms**:
- `RuntimeError: MPS backend out of memory`
- Training takes >30 minutes per epoch
- System becomes unresponsive during training
- GPU memory allocation failures

**Typical Limits**:
- MacBook M1/M2/M3: 8-20 GB unified memory
- Consumer GPUs: 8-12 GB (RTX 3060/3070)
- Workstation GPUs: 16-24 GB (RTX 3080/3090)

### Cloud Training Benefits

- **Larger datasets**: Train on 1,000+ patients instead of 100-200
- **Faster iteration**: 10x faster training on dedicated GPUs
- **Better performance**: More data → better C-index (0.65-0.75 vs 0.50-0.60)
- **Cost-effective**: Pay only for compute time (~$0.30-0.50/hour)

## Memory Requirements by Dataset Size

| Patients | Avg Visits | Vocab Size | Memory Needed | Recommended GPU |
|----------|-----------|------------|---------------|-----------------|
| 100 | 30 | 500 | 2-4 GB | Local MPS/CPU |
| 200 | 40 | 800 | 4-8 GB | RTX 3060 (12GB) |
| 500 | 50 | 1,500 | 8-12 GB | RTX 3080 (10GB) |
| 1,000 | 60 | 3,000 | 16-20 GB | RTX 3090 (24GB) |
| 2,000+ | 70 | 5,000 | 24-32 GB | RTX 4090 (24GB) or A100 (40GB) |

## RunPods Setup (Step-by-Step)

### 1. Create RunPods Account

1. Visit https://www.runpod.io/
2. Sign up with email or GitHub
3. Add payment method (credit card)
4. Add initial credits ($10-20 recommended)

### 2. Select GPU Pod

**Recommended GPUs** (as of 2026):

| GPU | VRAM | Price/hr | Best For |
|-----|------|----------|----------|
| RTX 3090 | 24 GB | $0.30 | Most cost-effective for our use case |
| RTX 4090 | 24 GB | $0.40 | Faster training, newer architecture |
| A100 (40GB) | 40 GB | $1.00 | Overkill for <2,000 patients |
| A100 (80GB) | 80 GB | $1.50 | Only for massive datasets (5,000+ patients) |

**Selection Process**:
1. Click "Deploy" → "GPU Pods"
2. Filter by GPU type (e.g., "RTX 3090")
3. Sort by price (lowest first)
4. Check "Secure Cloud" for reliability
5. Select pod with good uptime (>95%)

### 3. Configure Pod Template

**Option A: Use PyTorch Template** (Recommended)
```
Template: RunPod PyTorch 2.1
CUDA: 11.8 or 12.1
Python: 3.10+
Disk: 50 GB (sufficient for our data)
```

**Option B: Custom Docker Image**
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN pip install pandas numpy matplotlib seaborn scikit-learn tqdm jupyter
```

### 4. Upload Code and Data

**Method 1: Git Clone** (Recommended)
```bash
# SSH into pod
ssh root@<pod-ip> -p <port>

# Clone repository
git clone https://github.com/yourusername/ehr-sequencing.git
cd ehr-sequencing

# Install dependencies
pip install -e .
```

**Method 2: Jupyter Upload**
1. Open Jupyter interface (port 8888)
2. Upload notebook: `notebooks/02_survival_analysis/01_discrete_time_survival_lstm.ipynb`
3. Upload data directory: `data/synthea/large_cohort_1000/`

**Method 3: Cloud Storage**
```bash
# Download from S3/GCS
aws s3 sync s3://your-bucket/synthea-data ./data/synthea/large_cohort_1000/

# Or use wget for public URLs
wget https://your-storage.com/synthea-data.tar.gz
tar -xzf synthea-data.tar.gz
```

### 5. Install Dependencies

```bash
# If using git clone
cd ehr-sequencing
pip install -e .

# Or install manually
pip install pandas numpy matplotlib seaborn scikit-learn tqdm torch
```

### 6. Configure Notebook for Full Training

Open the notebook and modify cell 11:

```python
# Change from:
MAX_PATIENTS = 200  # Local testing

# To:
MAX_PATIENTS = None  # Full training on cloud GPU
```

### 7. Run Training

**Option A: Jupyter Notebook**
1. Open notebook in Jupyter
2. Run all cells (Cell → Run All)
3. Monitor progress in output

**Option B: Python Script**
```bash
# Convert notebook to script
jupyter nbconvert --to script 01_discrete_time_survival_lstm.ipynb

# Run as script
python 01_discrete_time_survival_lstm.py
```

**Option C: Screen Session** (for long training)
```bash
# Start screen session
screen -S training

# Run training
jupyter nbconvert --execute 01_discrete_time_survival_lstm.ipynb

# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

### 8. Monitor Training

**GPU Utilization**:
```bash
# Check GPU usage
nvidia-smi

# Watch in real-time
watch -n 1 nvidia-smi
```

**Expected Output**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 3090    Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   65C    P2   280W / 350W |  18432MiB / 24576MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

**Training Logs**:
```
Epoch 1/10: Train Loss=4.2089, Val Loss=3.5575, Val C-index=0.5234
Epoch 2/10: Train Loss=3.8123, Val Loss=3.2341, Val C-index=0.5891
Epoch 3/10: Train Loss=3.5234, Val Loss=3.0123, Val C-index=0.6234
...
```

### 9. Save Results

**Save Model Weights**:
```python
# In notebook, add cell:
torch.save(model.state_dict(), 'survival_lstm_1000patients.pth')
```

**Save Training History**:
```python
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history, f)
```

**Download Results**:
```bash
# From local machine
scp -P <port> root@<pod-ip>:/workspace/ehr-sequencing/survival_lstm_1000patients.pth ./
scp -P <port> root@<pod-ip>:/workspace/ehr-sequencing/training_history.pkl ./
```

### 10. Stop Pod

**Important**: Stop pod when done to avoid charges!

1. Go to RunPods dashboard
2. Click "Stop" on your pod
3. Verify pod is stopped (status: "Stopped")
4. Download any remaining files before terminating

## Cost Estimation

### Training Time Estimates

| Dataset | Epochs | Time per Epoch | Total Time | Cost (RTX 3090 @ $0.30/hr) |
|---------|--------|----------------|------------|----------------------------|
| 200 patients | 10 | 2 min | 20 min | $0.10 |
| 500 patients | 10 | 5 min | 50 min | $0.25 |
| 1,000 patients | 10 | 10 min | 100 min | $0.50 |
| 2,000 patients | 20 | 15 min | 300 min | $1.50 |

### Budget Planning

**Development Phase** (testing, debugging):
- Budget: $5-10
- Duration: 2-3 days
- Usage: Multiple short runs (10-30 min each)

**Training Phase** (final models):
- Budget: $10-20
- Duration: 1-2 days
- Usage: Few long runs (1-2 hours each)

**Production** (ongoing):
- Budget: $50-100/month
- Usage: Weekly retraining on new data

## Troubleshooting

### Out of Memory Errors

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `batch_size = 16` or `8`
2. Reduce model size: `embedding_dim=64, hidden_dim=128`
3. Enable gradient checkpointing
4. Use mixed precision training (FP16)

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    hazards = model(visit_codes, visit_mask, sequence_mask)
    loss = criterion(hazards, event_times, event_indicators, sequence_mask)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Slow Training

**Symptom**: <1 it/s, hours per epoch

**Solutions**:
1. Check GPU utilization: `nvidia-smi` (should be >80%)
2. Increase batch size if memory allows
3. Use DataLoader with `num_workers=4`
4. Pin memory: `DataLoader(..., pin_memory=True)`

### Connection Lost

**Symptom**: SSH/Jupyter disconnects during training

**Solutions**:
1. Use `screen` or `tmux` for persistent sessions
2. Save checkpoints every epoch
3. Enable auto-resume from last checkpoint

```python
# Save checkpoint every epoch
if (epoch + 1) % 1 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, f'checkpoint_epoch_{epoch+1}.pth')
```

### Data Transfer Issues

**Symptom**: Slow upload/download speeds

**Solutions**:
1. Compress data: `tar -czf data.tar.gz data/`
2. Use cloud storage (S3, GCS) as intermediate
3. Use `rsync` instead of `scp` for resumable transfers

```bash
# Resumable transfer
rsync -avz --progress -e "ssh -p <port>" \
  ./data/ root@<pod-ip>:/workspace/data/
```

## Best Practices

### 1. Start Small, Scale Up

```python
# First run: Test with subset
MAX_PATIENTS = 100  # Quick validation

# Second run: Medium dataset
MAX_PATIENTS = 500  # Verify scaling

# Final run: Full dataset
MAX_PATIENTS = None  # Production training
```

### 2. Use Version Control

```bash
# Track experiments
git checkout -b experiment/1000-patients-lstm
# ... make changes ...
git commit -m "Train on 1000 patients, C-index=0.68"
git tag v1.0-1000patients
```

### 3. Log Everything

```python
import logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

logging.info(f"Starting training with {len(sequences)} patients")
logging.info(f"Vocab size: {builder.vocabulary_size}")
# ... log metrics each epoch ...
```

### 4. Save Intermediate Results

```python
# Save every 5 epochs
if (epoch + 1) % 5 == 0:
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
    
# Save best model
if val_c_index > best_c_index:
    best_c_index = val_c_index
    torch.save(model.state_dict(), 'best_model.pth')
```

### 5. Monitor Costs

- Set spending alerts in RunPods dashboard
- Stop pods immediately after training
- Use spot instances for non-urgent training (50% cheaper)

## Alternative Cloud Providers

### Vast.ai
- **Pros**: Often cheaper than RunPods
- **Cons**: Less reliable, more setup required
- **Price**: RTX 3090 @ $0.20-0.30/hr

### Google Colab Pro
- **Pros**: Familiar interface, easy setup
- **Cons**: Session limits (12-24 hours), shared resources
- **Price**: $10/month for Pro

### Lambda Labs
- **Pros**: Dedicated GPUs, good for long training
- **Cons**: Higher minimum commitment
- **Price**: RTX 3090 @ $0.50/hr

### AWS/GCP/Azure
- **Pros**: Enterprise-grade, scalable
- **Cons**: Complex setup, expensive
- **Price**: V100 @ $2-3/hr, A100 @ $4-6/hr

## Summary

**For our survival LSTM training**:
- **Local**: 100-200 patients, testing/debugging
- **RunPods RTX 3090**: 500-1,000 patients, optimal cost/performance
- **RunPods RTX 4090**: 1,000-2,000 patients, faster training
- **RunPods A100**: 2,000+ patients or multi-GPU training

**Estimated total cost for complete project**: $10-30
- Development/testing: $5-10
- Final training runs: $5-10
- Hyperparameter tuning: $5-10

**Time to results**: 2-4 hours of actual training time spread over 1-2 days of development.
