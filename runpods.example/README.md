# RunPods Setup for ehr-sequencing

**This is an EXAMPLE** - Copy and customize for your setup.

---

## ⚠️ Important: This Directory is NOT Tracked

The `runpods/` directory contains user-specific configuration and is excluded from git via `.gitignore`.

This `runpods.example/` directory provides templates you can copy.

---

## 🚀 Quick Setup

### Step 1: Copy Example Directory

```bash
cd ~/work/ehr-sequencing
cp -r runpods.example runpods
```

### Step 2: Use Scripts

```bash
cd runpods/scripts
./runpod_ssh_manager.sh add ehr-sequencing
```

**Enter when prompted**:
- Hostname: `ssh.runpods.io` (from RunPods dashboard)
- Port: `12345` (from RunPods dashboard)
- Nickname: `a40-48gb` (or your GPU type)
- SSH Key: Press Enter for default

**Result**: SSH alias `runpod-ehr-sequencing-a40-48gb` created

### Step 3: Connect

```bash
ssh runpod-ehr-sequencing-a40-48gb
```

---

## 📋 What's Included

### Scripts (Self-Contained)

```
runpods.example/scripts/
├── runpod_ssh_manager.sh      # SSH config manager
├── quick_pod_setup.sh          # Automated setup
└── test_runpod_manager.sh      # Test suite
```

### Environment Configuration

```
runpods.example/environment-runpods-minimal.yml  # Minimal conda env for pods
```

---

## 🔧 Execution Model

### LOCAL (Your Machine)

Scripts run on **your local machine**:
- `runpod_ssh_manager.sh` - Configures SSH
- `quick_pod_setup.sh` - Automated setup
- `test_runpod_manager.sh` - Tests

**Modifies**: `~/.ssh/config` on your machine

### POD (RunPods Instance)

After SSH'ing to pod:

```bash
# 1. Setup GitHub SSH (for cloning private repos and pushing)
# See: runpods.example/docs/GITHUB_SSH_SETUP.md for detailed guide
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github -N ""
cat ~/.ssh/id_ed25519_github.pub  # Add this to https://github.com/settings/ssh/new

# 2. Configure SSH and Git
cat >> ~/.ssh/config <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  StrictHostKeyChecking no
EOF

git config --global user.name "pleiadian53"
git config --global user.email "pleiadian53@users.noreply.github.com"

# 3. Clone and setup environment
cd /workspace
git clone git@github.com:pleiadian53/ehr-sequencing.git
cd ehr-sequencing
mamba env create -f environment.yml
mamba activate ehr-sequencing
```

---

## 🔒 Privacy

### Why runpods/ is NOT in Git

- ❌ User-specific paths
- ❌ SSH configuration history
- ❌ Personal workflow customizations

### What IS Shared

- ✅ `runpods.example/` - This template
- ✅ Public setup guides

---

## 📚 Usage Workflow

### 1. Initial Setup (Once)

```bash
cd ~/work/ehr-sequencing
cp -r runpods.example runpods
cd runpods/scripts
```

### 2. Configure Pod Access (Per Pod)

```bash
./runpod_ssh_manager.sh add ehr-sequencing
```

### 3. Connect

```bash
ssh runpod-ehr-sequencing-a40-48gb
```

### 4. Setup Environment (On Pod)

```bash
cd /workspace
git clone <your-repo>
cd ehr-sequencing
mamba env create -f environment.yml
mamba activate ehr-sequencing
pip install -e .
```

### 5. Transfer Data (From Local)

```bash
rsync -avzP ~/work/ehr-sequencing/data/ \
  runpod-ehr-sequencing-a40-48gb:/workspace/data/
```

### 6. Start Work (On Pod)

```bash
tmux new -s training
cd /workspace/ehr-sequencing
mamba activate ehr-sequencing
python train.py
```

---

## 💡 Tips

- **Use tmux**: Sessions survive disconnection
- **Monitor GPU**: `watch -n 1 nvidia-smi`
- **Check costs**: RunPods dashboard
- **Terminate when done**: Only pay for compute time

---

## 📖 Documentation

Complete guides in `runpods.example/docs/`:

### Getting Started (in order)
1. **`POD_ENVIRONMENT_SETUP.md`** - Install Python environment on pod ⭐ Start here
2. **`IDE_JUPYTER_SETUP.md`** - Configure IDE and Jupyter notebooks
3. **`LOCAL_DEVELOPMENT_WORKFLOW.md`** - Sync code with rsync (recommended workflow)

### Optional / Advanced
- **`GITHUB_SSH_SETUP.md`** - Set up Git on pod (if using git instead of rsync)

---

**Created**: January 28, 2026  
**Status**: Template - Copy to `runpods/` and customize
