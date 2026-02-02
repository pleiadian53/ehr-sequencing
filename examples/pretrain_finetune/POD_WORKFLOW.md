# Pod Training Workflow Guide

Complete guide for training on ephemeral GPU pods and transferring results back to your local system.

## Overview

When training on cloud GPU pods (RunPods, Lambda Labs, etc.), you need to:
1. **Train** on the pod with automatic checkpointing
2. **Transfer** results back to local before pod terminates
3. **Analyze** results locally
4. **Version control** important artifacts (not all files)

---

## 1. Training on Pod

### Setup Pod
```bash
# SSH into pod
ssh root@<pod-ip>

# Clone repo
git clone https://github.com/yourusername/ehr-sequencing.git
cd ehr-sequencing

# Install dependencies
pip install -e .

# Verify GPU
nvidia-smi
```

### Run Training with Auto Resource Detection
```bash
# Auto-detects A40 and sets optimal parameters!
python examples/pretrain_finetune/train_behrt_demo.py --demo_data

# Or with realistic data
python examples/pretrain_finetune/train_behrt_demo.py --realistic_data

# Override specific parameters if needed
python examples/pretrain_finetune/train_behrt_demo.py \
    --demo_data \
    --batch_size 64 \
    --epochs 50
```

### Run in Background (Recommended)
```bash
# Use nohup to keep training if SSH disconnects
nohup python examples/pretrain_finetune/train_behrt_demo.py \
    --demo_data \
    > training.log 2>&1 &

# Check progress
tail -f training.log

# Or use tmux/screen for persistent sessions
tmux new -s training
python examples/pretrain_finetune/train_behrt_demo.py --demo_data
# Ctrl+B, D to detach
# tmux attach -t training to reattach
```

---

## 2. What Gets Generated

After training completes, you'll have:

```
experiments/behrt_large_mlm_lora16/
├── checkpoints/
│   ├── best_lora_weights.pt      # ⭐ ESSENTIAL - Best LoRA weights (~2-10MB)
│   ├── latest_lora_weights.pt    # Latest checkpoint (for resuming)
│   └── lora_weights_epoch_*.pt   # Periodic checkpoints
├── plots/
│   ├── loss_curve.png            # ⭐ ESSENTIAL - Training/val loss
│   ├── accuracy_curve.png        # ⭐ ESSENTIAL - Training/val accuracy
│   ├── macro_f1_curve.png        # ⭐ NEW - Macro F1 score
│   ├── weighted_f1_curve.png     # ⭐ NEW - Weighted F1 score
│   ├── top_5_accuracy_curve.png  # ⭐ NEW - Top-5 accuracy
│   └── perplexity_curve.png      # ⭐ NEW - Perplexity
├── logs/
│   └── metrics_history.json      # ⭐ ESSENTIAL - All metrics over time
├── hyperparameters.json          # ⭐ ESSENTIAL - Model config
├── metadata.json                 # Experiment metadata (timestamp, device, etc.)
├── summary.json                  # Machine-readable summary
└── SUMMARY.txt                   # ⭐ ESSENTIAL - Human-readable summary

nohup.out                         # Training logs (if using nohup)
```

---

## 3. Transfer Results to Local

### Option A: Transfer Essential Files Only (Recommended)

**What to transfer:**
- ✅ Best LoRA weights (~2-10MB)
- ✅ All plots (visualizations)
- ✅ Metrics history (for analysis)
- ✅ Hyperparameters & summary
- ❌ Skip: Intermediate checkpoints (large, not needed)
- ❌ Skip: Latest checkpoint (unless resuming training)

```bash
# On local machine
cd /path/to/ehr-sequencing

# Create experiment directory locally
mkdir -p experiments/behrt_large_mlm_lora16

# Transfer essential files
scp -r root@<pod-ip>:/workspace/ehr-sequencing/experiments/behrt_large_mlm_lora16/plots \
    experiments/behrt_large_mlm_lora16/

scp -r root@<pod-ip>:/workspace/ehr-sequencing/experiments/behrt_large_mlm_lora16/logs \
    experiments/behrt_large_mlm_lora16/

scp root@<pod-ip>:/workspace/ehr-sequencing/experiments/behrt_large_mlm_lora16/checkpoints/best_lora_weights.pt \
    experiments/behrt_large_mlm_lora16/checkpoints/

scp root@<pod-ip>:/workspace/ehr-sequencing/experiments/behrt_large_mlm_lora16/*.json \
    experiments/behrt_large_mlm_lora16/

scp root@<pod-ip>:/workspace/ehr-sequencing/experiments/behrt_large_mlm_lora16/SUMMARY.txt \
    experiments/behrt_large_mlm_lora16/

# Transfer training logs if using nohup
scp root@<pod-ip>:/workspace/ehr-sequencing/nohup.out \
    experiments/behrt_large_mlm_lora16/training.log
```

**Size estimate:** ~10-50MB total (mostly plots)

### Option B: Transfer Everything (If Resuming Training)

```bash
# Transfer entire experiment directory
scp -r root@<pod-ip>:/workspace/ehr-sequencing/experiments/behrt_large_mlm_lora16 \
    experiments/
```

**Size estimate:** ~50-200MB (includes all checkpoints)

### Option C: Use rsync (Most Efficient)

```bash
# Sync only new/changed files
rsync -avz --progress \
    root@<pod-ip>:/workspace/ehr-sequencing/experiments/behrt_large_mlm_lora16/ \
    experiments/behrt_large_mlm_lora16/

# Exclude intermediate checkpoints
rsync -avz --progress \
    --exclude 'checkpoints/lora_weights_epoch_*.pt' \
    --exclude 'checkpoints/latest_lora_weights.pt' \
    root@<pod-ip>:/workspace/ehr-sequencing/experiments/behrt_large_mlm_lora16/ \
    experiments/behrt_large_mlm_lora16/
```

---

## 4. Verify Transfer

```bash
# Check transferred files
ls -lh experiments/behrt_large_mlm_lora16/

# View summary
cat experiments/behrt_large_mlm_lora16/SUMMARY.txt

# Check plots exist
ls experiments/behrt_large_mlm_lora16/plots/

# Verify best checkpoint
ls -lh experiments/behrt_large_mlm_lora16/checkpoints/best_lora_weights.pt
```

---

## 5. Git Version Control

### What to Commit

**✅ DO commit:**
- Plots (visualizations are small and valuable)
- Summary files (SUMMARY.txt, summary.json)
- Hyperparameters (hyperparameters.json)
- Metrics history (logs/metrics_history.json)

**❌ DON'T commit:**
- Model checkpoints (too large, use Git LFS or external storage)
- nohup.out / training logs (too large, not reproducible)
- Intermediate checkpoints

### Update .gitignore

```bash
# Add to .gitignore if not already there
cat >> .gitignore << 'EOF'

# Experiment artifacts (large files)
experiments/*/checkpoints/*.pt
experiments/*/checkpoints/*.pth
experiments/**/nohup.out
experiments/**/training.log

# But DO track plots and summaries
!experiments/*/plots/*.png
!experiments/*/logs/metrics_history.json
!experiments/*/*.json
!experiments/*/*.txt
EOF
```

### Commit Results

```bash
# Stage essential files
git add experiments/behrt_large_mlm_lora16/plots/
git add experiments/behrt_large_mlm_lora16/logs/metrics_history.json
git add experiments/behrt_large_mlm_lora16/*.json
git add experiments/behrt_large_mlm_lora16/SUMMARY.txt

# Commit with descriptive message
git commit -m "Add training results: BEHRT large with LoRA rank 16

- Trained on A40 pod with demo data
- Final accuracy: 99.5%, F1: 0.965
- 5000 patients, 100 epochs, batch 128
- Plots and metrics included
- Best checkpoint: 2.1MB (LoRA only)"

git push
```

---

## 6. Store Model Checkpoints Separately

Model checkpoints are too large for Git. Use one of these options:

### Option A: Cloud Storage (Recommended)
```bash
# AWS S3
aws s3 cp experiments/behrt_large_mlm_lora16/checkpoints/best_lora_weights.pt \
    s3://your-bucket/ehr-sequencing/checkpoints/behrt_large_mlm_lora16_best.pt

# Google Cloud Storage
gsutil cp experiments/behrt_large_mlm_lora16/checkpoints/best_lora_weights.pt \
    gs://your-bucket/ehr-sequencing/checkpoints/
```

### Option B: Git LFS (For Smaller Checkpoints)
```bash
# Install Git LFS
git lfs install

# Track checkpoint files
git lfs track "experiments/*/checkpoints/*.pt"
git add .gitattributes

# Now you can commit checkpoints
git add experiments/behrt_large_mlm_lora16/checkpoints/best_lora_weights.pt
git commit -m "Add best LoRA checkpoint"
```

### Option C: Local Archive
```bash
# Create compressed archive
tar -czf behrt_large_mlm_lora16_checkpoints.tar.gz \
    experiments/behrt_large_mlm_lora16/checkpoints/

# Store in external drive or NAS
mv behrt_large_mlm_lora16_checkpoints.tar.gz /Volumes/ExternalDrive/
```

---

## 7. Analyze Results Locally

### View Training Curves
```bash
# Open plots
open experiments/behrt_large_mlm_lora16/plots/loss_curve.png
open experiments/behrt_large_mlm_lora16/plots/accuracy_curve.png
open experiments/behrt_large_mlm_lora16/plots/macro_f1_curve.png
```

### Load Metrics for Analysis
```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics history
with open('experiments/behrt_large_mlm_lora16/logs/metrics_history.json') as f:
    metrics = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(metrics)

# Custom analysis
df[['val_accuracy', 'val_macro_f1', 'val_top_5_accuracy']].plot()
plt.title('Validation Metrics Over Time')
plt.show()

# Find best epoch
best_epoch = df['val_loss'].idxmin()
print(f"Best epoch: {best_epoch}")
print(df.iloc[best_epoch])
```

### Load Checkpoint for Inference
```python
from ehrsequencing.models.behrt import BEHRT, BEHRTConfig
from ehrsequencing.models.lora import apply_lora_to_behrt, load_lora_weights

# Load config
with open('experiments/behrt_large_mlm_lora16/hyperparameters.json') as f:
    config_dict = json.load(f)

# Recreate model
config = BEHRTConfig.large(vocab_size=config_dict['vocab_size'])
model = BEHRT(config)
model = apply_lora_to_behrt(model, rank=config_dict['lora_rank'])

# Load trained weights
load_lora_weights(model, 'experiments/behrt_large_mlm_lora16/checkpoints/best_lora_weights.pt')

# Use for inference
model.eval()
# ... your inference code ...
```

---

## 8. Quick Reference Commands

### Before Terminating Pod
```bash
# 1. Verify training completed
tail experiments/behrt_large_mlm_lora16/SUMMARY.txt

# 2. Check file sizes
du -sh experiments/behrt_large_mlm_lora16/*

# 3. Compress if needed
tar -czf results.tar.gz experiments/behrt_large_mlm_lora16/
```

### Transfer Script (Save as `transfer_results.sh`)
```bash
#!/bin/bash
POD_IP=$1
EXPERIMENT_NAME=$2

if [ -z "$POD_IP" ] || [ -z "$EXPERIMENT_NAME" ]; then
    echo "Usage: ./transfer_results.sh <pod-ip> <experiment-name>"
    exit 1
fi

echo "Transferring results from pod..."

# Create local directory
mkdir -p experiments/$EXPERIMENT_NAME

# Transfer essential files
scp -r root@$POD_IP:/workspace/ehr-sequencing/experiments/$EXPERIMENT_NAME/plots \
    experiments/$EXPERIMENT_NAME/

scp -r root@$POD_IP:/workspace/ehr-sequencing/experiments/$EXPERIMENT_NAME/logs \
    experiments/$EXPERIMENT_NAME/

scp root@$POD_IP:/workspace/ehr-sequencing/experiments/$EXPERIMENT_NAME/checkpoints/best_lora_weights.pt \
    experiments/$EXPERIMENT_NAME/checkpoints/ 2>/dev/null || echo "No best checkpoint found"

scp root@$POD_IP:/workspace/ehr-sequencing/experiments/$EXPERIMENT_NAME/*.json \
    experiments/$EXPERIMENT_NAME/

scp root@$POD_IP:/workspace/ehr-sequencing/experiments/$EXPERIMENT_NAME/SUMMARY.txt \
    experiments/$EXPERIMENT_NAME/

echo "Transfer complete! Results in experiments/$EXPERIMENT_NAME/"
```

**Usage:**
```bash
chmod +x transfer_results.sh
./transfer_results.sh 123.45.67.89 behrt_large_mlm_lora16
```

---

## 9. Troubleshooting

### "Permission denied" during scp
```bash
# Check SSH key
ssh root@<pod-ip> "ls -la ~/.ssh/"

# Or use password authentication
scp -o PreferredAuthentications=password ...
```

### "Connection timed out"
```bash
# Check pod is still running
# Verify IP address hasn't changed
# Try with verbose output
scp -v root@<pod-ip>:...
```

### "No space left on device" on pod
```bash
# Check disk usage
df -h

# Clean up old experiments
rm -rf experiments/old_experiment_*

# Or compress and transfer incrementally
tar -czf results_part1.tar.gz experiments/exp1/
scp results_part1.tar.gz local:~/
rm results_part1.tar.gz
```

### Missing files after transfer
```bash
# Verify on pod first
ssh root@<pod-ip> "ls -R experiments/behrt_large_mlm_lora16/"

# Check transfer logs
scp -v ... 2>&1 | tee transfer.log
```

---

## 10. Best Practices

### ✅ DO
1. **Transfer immediately** after training completes
2. **Verify transfer** before terminating pod
3. **Commit plots and summaries** to Git
4. **Store checkpoints** in cloud storage or Git LFS
5. **Document experiment** in commit message
6. **Use rsync** for efficient transfers
7. **Keep training logs** (nohup.out) temporarily

### ❌ DON'T
1. **Don't terminate pod** before transferring
2. **Don't commit large checkpoints** to Git (use LFS)
3. **Don't lose training logs** - they're useful for debugging
4. **Don't transfer everything** - be selective
5. **Don't forget metadata** - hyperparameters are crucial

---

## Summary

**Minimal workflow:**
```bash
# 1. Train on pod (auto-detects resources!)
python train_behrt_demo.py --demo_data

# 2. Transfer results (on local)
./transfer_results.sh <pod-ip> behrt_large_mlm_lora16

# 3. Commit to Git
git add experiments/behrt_large_mlm_lora16/{plots,logs,*.json,*.txt}
git commit -m "Add training results: BEHRT large LoRA"

# 4. Store checkpoint separately
aws s3 cp experiments/behrt_large_mlm_lora16/checkpoints/best_lora_weights.pt \
    s3://bucket/checkpoints/
```

**That's it!** Your results are safely stored locally and version controlled.
