# RunPod Setup Guide

**Date:** January 20, 2026  
**Focus:** Setting up ehr-sequencing on RunPod GPU instances for training

---

## Overview

This guide covers setting up the ehr-sequencing project on RunPod GPU instances. For local development, see [local-setup.md](local-setup.md).

**Recommended GPUs:**
- **A10 (24GB)**: Medium models, good price/performance
- **RTX 4090 (24GB)**: Fast training, good for medium models
- **A40 (48GB)**: Large models, production training
- **A100 (80GB)**: Largest models, fastest training

---

## Prerequisites

- RunPod account with credits
- SSH key configured on your local machine
- Basic familiarity with Linux/bash

---

## Step 1: Deploy RunPod Instance

### 1.1 Select Template

1. Go to [RunPod](https://www.runpod.io/)
2. Click **Deploy** → **GPU Pods**
3. Select GPU:
   - **A10 24GB**: ~$0.50/hr (recommended for medium models)
   - **RTX 4090 24GB**: ~$0.60/hr
   - **A40 48GB**: ~$1.00/hr (recommended for large models)
   - **A100 80GB**: ~$2.00/hr

### 1.2 Configure Pod

- **Container Image**: `runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04`
- **Container Disk**: 50GB minimum
- **Volume**: 100GB (optional, for persistent storage)
- **Expose HTTP/TCP Ports**: Enable (for Jupyter, TensorBoard)

### 1.3 Deploy and Connect

1. Click **Deploy**
2. Wait for pod to start (~1-2 minutes)
3. Copy SSH command from **Connect** → **SSH over exposed TCP**

```bash
# Example SSH command
ssh root@69.30.85.30 -p 22084 -i ~/.ssh/id_ed25519
```

---

## Step 2: Initial Setup

### 2.1 Install Miniforge

```bash
# Navigate to temporary location
cd /tmp

# Download Miniforge
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# Install
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3

# Clean up
rm Miniforge3-$(uname)-$(uname -m).sh

# Initialize shell
~/miniforge3/bin/conda init bash
source ~/.bashrc
```

### 2.2 Verify GPU

```bash
# Check GPU
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.1   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA A40          Off  | 00000000:00:05.0 Off |                    0 |
# |  0%   30C    P0    72W / 300W |      0MiB / 46068MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

---

## Step 3: Clone and Setup Project

### 3.1 Clone Repository

```bash
cd /workspace

# Clone via HTTPS
git clone https://github.com/YOUR_USERNAME/ehr-sequencing.git

# Or via SSH (if SSH key configured)
git clone git@github.com:YOUR_USERNAME/ehr-sequencing.git

cd ehr-sequencing
```

### 3.2 Create Environment

```bash
# Create environment
mamba env create -f environment.yml

# Activate
mamba activate ehrsequencing

# Install package
pip install -e .
```

### 3.3 Verify Installation

```bash
# Check PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Expected output:
# PyTorch: 2.1.0
# CUDA available: True
# GPU: NVIDIA A40

# Check package
python -c "import ehrsequencing; print(ehrsequencing.__version__)"
```

---

## Step 4: Transfer Data

### Option A: rsync from Local Machine

```bash
# From your local machine
rsync -avz --progress ./data/ root@POD_IP:/workspace/ehr-sequencing/data/ -e "ssh -p PORT"

# Example:
rsync -avz --progress ./data/ root@69.30.85.30:/workspace/ehr-sequencing/data/ -e "ssh -p 22084"
```

### Option B: Download from Cloud Storage

```bash
# AWS S3
pip install awscli
aws configure
aws s3 sync s3://your-bucket/data/ /workspace/ehr-sequencing/data/

# Google Cloud Storage
pip install gcloud
gcloud auth login
gsutil -m rsync -r gs://your-bucket/data/ /workspace/ehr-sequencing/data/
```

### Option C: Generate Synthea Data

```bash
# Install Synthea
cd /workspace
git clone https://github.com/synthetichealth/synthea.git
cd synthea

# Generate data
./run_synthea -p 10000

# Copy to project
cp output/csv/*.csv /workspace/ehr-sequencing/data/synthea/
```

---

## Step 5: Training Configuration

### 5.1 Select Model Config

```bash
# For A10 24GB: Use medium config
python -c "
from ehrsequencing.models.configs import get_model_config
config = get_model_config('medium', 'lstm')
print(f'Model: {config.model_type}')
print(f'Parameters: ~{config.total_params_millions}M')
print(f'Memory: ~{config.memory_estimate_gb(dtype_bytes=2)}GB')
print(f'Batch size: {config.batch_size}')
"

# For A40 48GB: Use large config
config = get_model_config('large', 'lstm')
```

### 5.2 Test Small Run

```bash
# Quick test to verify everything works
python scripts/train_disease_progression.py \
    --size small \
    --model-type lstm \
    --epochs 5 \
    --data-path data/synthea/sequences.pt
```

---

## Step 6: Full Training

### 6.1 Training Script

```bash
# Medium model on A10 24GB
python scripts/train_disease_progression.py \
    --size medium \
    --model-type lstm \
    --epochs 100 \
    --data-path data/synthea/sequences.pt \
    --checkpoint-dir checkpoints/ \
    --experiment-name "ckd_progression_medium_lstm" \
    --log-interval 10 \
    --save-interval 10

# Large model on A40 48GB
python scripts/train_disease_progression.py \
    --size large \
    --model-type lstm \
    --epochs 100 \
    --data-path data/synthea/sequences.pt \
    --checkpoint-dir checkpoints/ \
    --experiment-name "ckd_progression_large_lstm"
```

### 6.2 Monitor Training

**Option A: TensorBoard**

```bash
# Start TensorBoard
tensorboard --logdir=runs/ --port=6006 --bind_all

# Access via RunPod port forwarding
# Go to RunPod dashboard → Your pod → Connect → HTTP Service
# Add port 6006
```

**Option B: Watch Logs**

```bash
# Tail training logs
tail -f logs/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### 6.3 Background Training (tmux)

```bash
# Install tmux
apt-get update && apt-get install -y tmux

# Start tmux session
tmux new -s training

# Run training
python scripts/train_disease_progression.py --size medium --epochs 100

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
# Kill session: tmux kill-session -t training
```

---

## Step 7: Download Results

### 7.1 Download Checkpoints

```bash
# From local machine
rsync -avz --progress root@POD_IP:/workspace/ehr-sequencing/checkpoints/ ./checkpoints/ -e "ssh -p PORT"
```

### 7.2 Download Logs

```bash
# Download TensorBoard logs
rsync -avz --progress root@POD_IP:/workspace/ehr-sequencing/runs/ ./runs/ -e "ssh -p PORT"
```

---

## Performance Benchmarks

### Training Speed (Medium LSTM, 10K patients)

| GPU | Batch Size | Time/Epoch | Memory Usage |
|-----|-----------|------------|--------------|
| A10 24GB | 32 | ~2 min | ~8GB |
| RTX 4090 24GB | 32 | ~1.5 min | ~8GB |
| A40 48GB | 64 | ~1 min | ~16GB |
| A100 80GB | 128 | ~30 sec | ~32GB |

### Cost Estimates (100 epochs)

| GPU | Time | Cost/hr | Total Cost |
|-----|------|---------|------------|
| A10 24GB | ~3.5 hrs | $0.50 | ~$1.75 |
| RTX 4090 24GB | ~2.5 hrs | $0.60 | ~$1.50 |
| A40 48GB | ~1.7 hrs | $1.00 | ~$1.70 |
| A100 80GB | ~0.8 hrs | $2.00 | ~$1.60 |

---

## Best Practices

### 1. Use Persistent Volumes

```bash
# Mount volume at /workspace
# Data persists across pod restarts
# Costs ~$0.10/GB/month
```

### 2. Save Checkpoints Frequently

```python
# In training script
if epoch % 10 == 0:
    torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch}.pt')
```

### 3. Monitor Costs

```bash
# Check pod uptime
uptime

# Stop pod when not training
# RunPod dashboard → Stop Pod
```

### 4. Use Spot Instances (if available)

- 50-70% cheaper than on-demand
- Can be interrupted (use checkpointing)
- Good for non-critical training

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python scripts/train_disease_progression.py --batch-size 16

# Enable gradient checkpointing
# Edit config: config.use_checkpoint = True

# Use smaller model
python scripts/train_disease_progression.py --size small
```

### Slow Training

```bash
# Check GPU utilization
nvidia-smi

# Enable mixed precision
# config.mixed_precision = True

# Increase batch size (if memory allows)
python scripts/train_disease_progression.py --batch-size 64
```

### Connection Lost

```bash
# Use tmux to keep training running
tmux new -s training
python scripts/train_disease_progression.py ...

# Detach and close SSH
# Training continues in background
```

---

## SSH Configuration (Local Machine)

Add to `~/.ssh/config`:

```bash
# RunPod Instance for ehr-sequencing
Host runpod-ehr
    # Update these from RunPod dashboard
    HostName 69.30.85.30
    Port 22084
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 5
    Compression yes
```

Connect with:

```bash
ssh runpod-ehr
```

---

## Jupyter Notebook Access

### Setup

```bash
# On RunPod
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Get token from output
# http://0.0.0.0:8888/?token=YOUR_TOKEN
```

### Access

1. Go to RunPod dashboard
2. Your pod → Connect → HTTP Service
3. Add port 8888
4. Click link, paste token

---

## Related Documentation

- [INSTALL.md](../../INSTALL.md) - Quick start
- [Local Setup](local-setup.md) - M1 MacBook development
- [Resource-Aware Models](../../docs/implementation/resource-aware-models.md) - Model configs
- [Troubleshooting](troubleshooting.md) - Common issues

---

**Last Updated:** January 20, 2026
