# Pod Environment Setup Guide

Complete guide for setting up the Python environment on a new RunPods instance for EHR sequencing model training.

---

## 🎯 Overview

After syncing your code to `/workspace/ehr-sequencing/` via rsync, you need to:

1. Install Miniforge (includes mamba package manager)
2. Create the conda environment
3. Install PyTorch with CUDA support
4. Install the project in editable mode

**Estimated time**: 10-15 minutes

---

## ⚡ Quick Setup (Copy-Paste)

SSH to your pod and run these commands:

```bash
# 1. Install or Activate Miniforge
cd /workspace

# If Miniforge already exists from another project, just activate it:
if [ -f /workspace/miniforge3/bin/mamba ]; then
    echo "Miniforge already installed, activating..."
    eval "$(/workspace/miniforge3/bin/conda shell.bash hook)"
else
    echo "Installing Miniforge..."
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3
    /workspace/miniforge3/bin/conda init bash
    source ~/.bashrc
fi

# Verify mamba works
mamba --version

# 2. Create environment
cd /workspace/ehr-sequencing

# If environment already exists, remove it first:
# mamba env remove -n ehrsequencing

# Option A: Minimal (training only, ~5-7 min)
mamba env create -f runpods.example/environment-runpods-minimal.yml

# Option B: Full (includes Jupyter, ~7-10 min)
# mamba env create -f environment.yml

# 3. Activate environment (if this fails, see Step 3 troubleshooting)
eval "$(mamba shell hook --shell bash)"  # Needed if shell not initialized
mamba activate ehrsequencing

# 4. Install PyTorch with CUDA support (for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install project in editable mode
pip install -e .

# 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python -c "import ehrsequencing; print('EHR Sequencing package: OK')"
```

✅ **Done!** Your environment is ready for EHR sequence modeling.

---

## 📋 Step-by-Step Instructions

### Prerequisites

- ✅ Pod is running
- ✅ You can SSH to the pod
- ✅ Code synced to `/workspace/ehr-sequencing/` via rsync
- ✅ Internet connectivity on pod

### Step 1: Install or Activate Miniforge

**Why Miniforge?**
- Includes `mamba` (faster than conda)
- Free, open-source
- Works well with GPUs
- Standard for scientific computing

#### Check if Miniforge Already Exists

If you've used this pod for other projects (e.g., meta-spliceai, agentic-spliceai), Miniforge might already be installed:

```bash
# Check if Miniforge exists
ls -la /workspace/miniforge3/bin/mamba
```

**If it exists**, skip to "Activate Existing Installation" below.

**If it doesn't exist**, continue with installation:

#### Fresh Installation

```bash
# SSH to your pod
ssh <your-pod-ssh-alias>

# Navigate to workspace
cd /workspace

# Download Miniforge installer
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# Install (non-interactive, installs to /workspace/miniforge3)
bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3
```

**Explanation of flags**:
- `-b`: Batch mode (no prompts)
- `-p /workspace/miniforge3`: Install to persistent workspace

#### Activate Existing Installation

If Miniforge is already installed but `mamba` command doesn't work:

```bash
# Initialize conda in your current shell
/workspace/miniforge3/bin/conda init bash

# Option A: Start a new terminal (recommended)
exit
ssh <your-pod-ssh-alias>

# Option B: Activate in current terminal
source ~/.bashrc
# OR if that doesn't work:
eval "$(/workspace/miniforge3/bin/conda shell.bash hook)"
```

**Verify mamba is working**:
```bash
which mamba
# Should show: /workspace/miniforge3/bin/mamba

mamba --version
# Should show: mamba x.x.x, conda x.x.x
```

✅ **Miniforge ready!**

---

### Step 2: Create Environment

You have two options:

#### Option A: Minimal (Recommended for Training-Only)

Fastest setup, includes only essential packages for model training:

```bash
cd /workspace/ehr-sequencing
mamba env create -f runpods.example/environment-runpods-minimal.yml
```

**What's included**:
- Python 3.10
- Core ML: PyTorch (via pip), scikit-learn
- Data Science: pandas, numpy, scipy
- EHR tools: lifelines, gensim, umap-learn, biopython
- Transformers: HuggingFace transformers, einops
- Visualization: matplotlib, plotly, seaborn
- Experiment tracking: wandb, tensorboard

**What's NOT included**:
- Jupyter notebooks (add: `pip install jupyter ipykernel notebook`)
- Dev tools (pytest, black, ruff - not needed on pod)

**Install time**: ~5-7 minutes

#### Option B: Full Environment (For Development + Training)

Includes Jupyter and dev tools - useful if you'll do analysis on pod:

```bash
cd /workspace/ehr-sequencing
mamba env create -f environment.yml
```

**Additionally includes**:
- Jupyter notebooks + ipykernel
- Dev tools: pytest, black, ruff, mypy

**Install time**: ~7-10 minutes

**Note**: The full `environment.yml` is already quite lean. The difference is only ~7 packages.

#### Recommendation

- **Use Minimal** if you only need to train and will analyze results locally
- **Use Full** if you want to run Jupyter notebooks on the pod for quick debugging

---

### Step 3: Activate Environment

```bash
# If you get "Shell not initialized" error, run this first:
eval "$(mamba shell hook --shell bash)"

# Then activate
mamba activate ehrsequencing
```

You should see your prompt change to:
```bash
(ehrsequencing) root@runpods:~$
```

**If activation fails with "Shell not initialized"**:
This happens when your current shell wasn't properly initialized with mamba. Quick fix:
```bash
# One-time fix for current session
eval "$(mamba shell hook --shell bash)"
mamba activate ehrsequencing

# OR start a new shell (picks up ~/.bashrc initialization)
bash
mamba activate ehrsequencing
```

**To make it activate automatically** in new shells:
```bash
echo 'mamba activate ehrsequencing' >> ~/.bashrc
```

---

### Step 4: Install PyTorch with CUDA

**IMPORTANT**: PyTorch from conda doesn't always match your GPU. Install from PyTorch directly:

```bash
# For CUDA 12.1 (common on RunPods)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For different CUDA versions**:
```bash
# Check your CUDA version first
nvidia-smi

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Verify PyTorch + GPU**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output**:
```
PyTorch: 2.x.x+cu121
CUDA available: True
GPU count: 1
GPU name: NVIDIA A40
```

✅ **PyTorch configured for GPU!**

---

### Step 5: Install Project

Install the project in **editable mode** (changes to code reflect immediately):

```bash
cd /workspace/ehr-sequencing
pip install -e .
```

**What this does**:
- Installs project as a package
- Editable mode (`-e`): changes reflect without reinstall
- Makes imports work: `from ehrsequencing import ...`

**Verify installation**:
```bash
python -c "import ehrsequencing; print('EHR Sequencing installed successfully!')"
```

---

### Step 6: Optional Dependencies

#### For Survival Analysis
```bash
pip install lifelines>=0.27.0
```

#### For NLP & Embeddings
```bash
# Word2Vec and doc2vec embeddings
pip install gensim>=4.3.0

# Additional NLP tools
pip install spacy
python -m spacy download en_core_web_sm
```

#### For Jupyter Notebooks (Local Analysis)
```bash
pip install jupyter ipykernel notebook
```

#### For Experiment Tracking
```bash
# Weights & Biases (already in minimal env)
wandb login

# TensorBoard (already in minimal env)
tensorboard --logdir=runs
```

#### For UMAP Dimensionality Reduction
```bash
pip install umap-learn>=0.5.0
```

---

## 🔧 Troubleshooting

### Problem: `mamba: command not found`

**Cause**: Miniforge installed but not active in current shell (common when reusing pods from other projects)

**Diagnosis**:
```bash
# Check if Miniforge exists
ls -la /workspace/miniforge3/bin/mamba
# If this shows the file, Miniforge is installed but not activated
```

**Solution**:
```bash
# Initialize conda for bash
/workspace/miniforge3/bin/conda init bash

# Then either:
# Option A: Start a new terminal (cleanest)
exit
ssh <your-pod-ssh-alias>

# Option B: Activate in current terminal
source ~/.bashrc

# Option C: If Option B doesn't work (most reliable)
eval "$(/workspace/miniforge3/bin/conda shell.bash hook)"

# Verify it works
mamba --version
```

**Why this happens**: When you SSH into a pod, your shell loads `~/.bashrc`. If Miniforge was installed in a different terminal session, the current session doesn't know about it until you initialize or reload the shell configuration.

---

### Problem: `mamba activate` fails with "Shell not initialized"

**Error**: `'mamba' is running as a subprocess and can't modify the parent shell. Shell not initialized`

**Cause**: Your current shell session isn't hooked into mamba

**Solution**:
```bash
# Quick fix - hook current shell into mamba
eval "$(mamba shell hook --shell bash)"

# Now activation will work
mamba activate ehrsequencing
```

**Alternative - Start a fresh shell**:
```bash
# Exit and reconnect, OR just start a new bash session
bash
mamba activate ehrsequencing
```

**Why this happens**: `mamba activate` needs to modify your shell's environment variables. This requires the shell to be "initialized" with mamba's shell functions. The `eval` command loads these functions into your current session.

### Problem: `CUDA not available` even with GPU

**Possible causes**:

1. **Wrong PyTorch build**
   ```bash
   # Check PyTorch build
   python -c "import torch; print(torch.__version__)"
   # Should show: 2.x.x+cu121 (or cu118, cu124)
   # If it shows: 2.x.x+cpu, you installed CPU-only version
   
   # Fix: Reinstall with CUDA
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **CUDA driver mismatch**
   ```bash
   nvidia-smi
   # Check CUDA version at top right
   # Install matching PyTorch version
   ```

3. **Environment issue**
   ```bash
   # Check environment
   echo $CUDA_VISIBLE_DEVICES
   # Should be empty or show GPU IDs (0,1,2...)
   
   # If it shows -1 or wrong values, unset it
   unset CUDA_VISIBLE_DEVICES
   ```

### Problem: Environment already exists

**Error**: `Found conda-prefix at '/workspace/miniforge3/envs/ehrsequencing'. Overwrite?: [y/N]` followed by `cannot remove: Directory not empty`

**Cause**: An `ehrsequencing` environment already exists (possibly from a previous attempt or another project)

**Solution**:
```bash
# Remove the existing environment
mamba env remove -n ehrsequencing

# Verify it's gone
mamba env list

# Create fresh environment
cd /workspace/ehr-sequencing
mamba env create -f environment.yml
```

### Problem: Corrupted environment - cannot remove

**Error**: `critical libmamba could not load prefix data: failed to run python command` when trying to remove environment

**Cause**: Environment is corrupted (Python interpreter broken or missing)

**Solution** - Manually delete the environment directory:
```bash
# Force remove the corrupted environment directory
rm -rf /workspace/miniforge3/envs/ehrsequencing

# Verify it's gone
ls -la /workspace/miniforge3/envs/
# Should NOT show ehrsequencing

# Clean conda cache (optional but recommended)
mamba clean --all -y

# Now create fresh environment
cd /workspace/ehr-sequencing
mamba env create -f environment.yml
```

**Note**: Manual deletion is safe - you're just removing a directory. The environment will be recreated cleanly.

**If `rm -rf` still fails with "Directory not empty"**:
```bash
# Try more aggressive removal
find /workspace/miniforge3/envs/ehrsequencing -delete

# OR fix permissions first, then remove
chmod -R 777 /workspace/miniforge3/envs/ehrsequencing
rm -rf /workspace/miniforge3/envs/ehrsequencing
```

**If deletion still fails (FUSE filesystem + hard links issue on RunPods)**:

RunPods uses a FUSE network filesystem which sometimes has issues with `rm -rf` when hard links are present.

```bash
# Option A: Use rsync to delete (handles FUSE better)
mkdir /tmp/empty
rsync -a --delete /tmp/empty/ /workspace/miniforge3/envs/ehrsequencing/
rm -rf /workspace/miniforge3/envs/ehrsequencing

# Option B: Just rename it out of the way
mv /workspace/miniforge3/envs/ehrsequencing /workspace/miniforge3/envs/ehrsequencing.OLD

# Option C: Use a different environment name (fastest)
cd /workspace/ehr-sequencing
sed 's/name: ehrsequencing/name: ehrseq/' environment.yml > environment-temp.yml
mamba env create -f environment-temp.yml
mamba activate ehrseq  # Note: use 'ehrseq' instead of 'ehrsequencing'
```

**Recommended**: Use Option C - it's fastest and avoids filesystem issues entirely.

### Problem: Environment creation fails

**Common issues**:

1. **Dependency conflicts**
   ```bash
   # Try with --force-reinstall
   mamba env create -f environment.yml --force
   
   # Or remove and retry
   mamba env remove -n ehrsequencing
   mamba env create -f runpods.example/environment-runpods-minimal.yml
   ```

2. **Network timeout**
   ```bash
   # Increase timeout
   conda config --set remote_read_timeout_secs 600
   
   # Retry
   mamba env create -f runpods.example/environment-runpods-minimal.yml
   ```

3. **Disk space**
   ```bash
   df -h /workspace
   # Ensure you have at least 5GB free
   
   # Clean conda cache if needed
   mamba clean --all -y
   ```

### Problem: Import errors after installation

```bash
# Verify environment is activated
conda env list
# Should show * next to ehrsequencing

# Reinstall project
cd /workspace/ehr-sequencing
pip install -e . --force-reinstall

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
# Should include /workspace/ehr-sequencing
```

### Problem: Slow environment creation

**Solutions**:
```bash
# 1. Use minimal environment (faster)
mamba env create -f runpods.example/environment-runpods-minimal.yml

# 2. Use mamba instead of conda (10x faster)
mamba env create -f environment.yml

# 3. Use libmamba solver with conda
conda config --set solver libmamba
conda env create -f environment.yml
```

---

## 📊 Environment Comparison

| Aspect | Minimal (RunPods) | Full (environment.yml) |
|--------|-------------------|------------------------|
| **Install time** | 5-7 minutes | 7-10 minutes |
| **Disk space** | ~2.5 GB | ~3 GB |
| **Python version** | 3.10 | 3.10 |
| **PyTorch** | ✅ (via pip + CUDA) | ✅ (via conda) |
| **Transformers** | ✅ | ✅ |
| **Jupyter** | ❌ (add if needed) | ✅ |
| **Survival tools** | ✅ (lifelines) | ✅ (lifelines) |
| **NLP tools** | ✅ (transformers, gensim) | ✅ (transformers, gensim) |
| **Dev tools** | ❌ | ✅ (pytest, black, ruff) |
| **Best for** | Training only | Training + pod analysis |

**Note**: Both are EHR-focused and quite lean. The difference is only ~7 packages (Jupyter + dev tools).

---

## 🚀 Quick Environment Tests

After setup, verify everything works:

### Test 1: Python Environment
```bash
python -c "import sys; print(f'Python: {sys.version}')"
```

### Test 2: PyTorch + GPU
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Test 3: NumPy + Scientific Stack
```bash
python -c "import numpy as np; import pandas as pd; import sklearn; print('Scientific stack: OK')"
```

### Test 4: Transformers & NLP
```bash
python -c "from transformers import AutoModel; print('Transformers: OK')"
```

### Test 5: Project Installation
```bash
python -c "import ehrsequencing; from ehrsequencing.models import SurvivalLSTM; print('EHR Sequencing: OK')"
```

### Test 6: CUDA Device Check
```bash
python -c "import torch; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'Device: {device}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB' if torch.cuda.is_available() else 'CPU only')"
```

**All tests pass? You're ready to train EHR sequence models! 🎉**

---

## 💡 Best Practices

### 1. Use Minimal Environment for RunPods
- Faster setup
- Less disk usage
- Sufficient for training

### 2. Install PyTorch Separately
- Better CUDA compatibility
- More control over version
- Avoid conda PyTorch issues

### 3. Keep Environment Files Updated
- Commit environment changes to git (locally)
- Sync to pod via rsync
- Recreate environment if major changes

### 4. Monitor Disk Space
```bash
# Check space before creating environment
df -h /workspace

# Clean conda cache after setup
mamba clean --all -y
```

### 5. Document Custom Installations
If you install additional packages:
```bash
# Save environment state
mamba env export > environment-snapshot.yml
```

---

## 🔄 Updating Environment

### Add New Package
```bash
# Activate environment
mamba activate ehrsequencing

# Install package
pip install new-package
# or
mamba install -c conda-forge new-package

# Optional: Update environment file locally
# (on your local machine, add to environment.yml)
```

### Recreate Environment
```bash
# Remove old environment
mamba env remove -n ehrsequencing

# Sync latest code from local
# (on local machine: sync-to-pod)

# Create fresh environment
cd /workspace/ehr-sequencing
mamba env create -f runpods.example/environment-runpods-minimal.yml
mamba activate ehrsequencing
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

---

## 📝 Summary Checklist

After setup is complete, verify:

- [ ] Miniforge installed (`which mamba` works)
- [ ] Environment created (`mamba env list` shows ehrsequencing)
- [ ] Environment activated (prompt shows `(ehrsequencing)`)
- [ ] PyTorch with CUDA installed (`torch.cuda.is_available()` returns True)
- [ ] Project installed (`import ehrsequencing` works)
- [ ] GPU detected (`nvidia-smi` shows your GPU)
- [ ] All import tests pass (see Test 1-6 above)

**Estimated total time**: 10-15 minutes  
**Frequency**: Once per new pod  
**Result**: Ready for EHR sequence model training 🚀

---

## 🔗 Related Documentation

- **IDE & Jupyter Setup**: `IDE_JUPYTER_SETUP.md` - Configure notebooks and IDE
- **Pod SSH Setup**: `runpod_ssh_manager.sh` - Connecting to pod
- **Code Sync**: `RSYNC_QUICK_REFERENCE.md` - Syncing code/results
- **GitHub Setup** (if needed): `GITHUB_SSH_SETUP.md` - Git on pod
- **Local Development**: `LOCAL_DEVELOPMENT_WORKFLOW.md` - Recommended workflow

---

## 🔧 IDE Setup (Jupyter/VS Code/Cursor)

After creating the environment, you need to configure your IDE to use it.

**Quick setup**:
```bash
# Install Jupyter kernel
mamba run -n ehrsequencing python -m ipykernel install --user --name=ehrsequencing --display-name="EHR Sequencing (Python 3.10)"

# Get interpreter path for IDE
echo "/workspace/miniforge3/envs/ehrsequencing/bin/python"
```

**Then in your IDE**:
- **For notebooks**: Select "EHR Sequencing (Python 3.10)" as kernel
- **For Python files**: Use interpreter: `/workspace/miniforge3/envs/ehrsequencing/bin/python`

📘 **For detailed instructions, see**: [`IDE_JUPYTER_SETUP.md`](./IDE_JUPYTER_SETUP.md)

---

## 🎯 Next Steps

Once environment is set up:

1. **Verify data is available**:
   ```bash
   ls -lh /workspace/data/
   # Should contain your EHR/Synthea data
   ```

2. **Test example training scripts**:
   ```bash
   cd /workspace/ehr-sequencing
   python examples/train_lstm_baseline.py --help
   python examples/train_survival_lstm.py --help
   ```

3. **Start training**:
   ```bash
   # In tmux (recommended - survives disconnection)
   tmux new -s training
   cd /workspace/ehr-sequencing
   
   # Example: Train LSTM baseline
   python examples/train_lstm_baseline.py --data_path /workspace/data/synthea
   
   # Example: Train survival model
   python examples/train_survival_lstm.py --data_path /workspace/data/synthea --epochs 50
   
   # Ctrl+B, D to detach
   ```

4. **Monitor training**:
   ```bash
   # Reattach to tmux
   tmux attach -t training
   
   # Or monitor with tensorboard
   tensorboard --logdir=runs --host=0.0.0.0 --port=6006
   
   # Or check wandb (if configured)
   # https://wandb.ai/your-username/ehr-sequencing
   ```

5. **Download results** (from local machine):
   ```bash
   # Sync trained models and outputs
   rsync -avzP <pod-ssh-alias>:/workspace/ehr-sequencing/checkpoints/ \
     ~/work/ehr-sequencing/checkpoints/
   
   rsync -avzP <pod-ssh-alias>:/workspace/ehr-sequencing/runs/ \
     ~/work/ehr-sequencing/runs/
   ```

---

**Created**: January 28, 2026  
**Version**: 1.0.0  
**Applies to**: All RunPods instances with NVIDIA GPUs for EHR sequence modeling
