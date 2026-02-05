# 1. Activate mamba (if not already active)
eval "$(/workspace/miniforge3/bin/conda shell.bash hook)"

# 2. Create the environment
cd /workspace/ehr-sequencing
mamba env create -f environment.yml
# Type "Y" when prompted

# 3. After it completes, activate it
mamba activate ehrsequencing

# 4. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install the project
pip install -e .

# 6. Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import ehrsequencing; print('EHR Sequencing: OK')"



---

Great question! The answer is **it depends on what you mean by "retain this drive"**:

## ❌ If You TERMINATE the Pod

When you **terminate** the pod:
- `/workspace` (FUSE network drive) = ✅ **Persists**
- `/root/` (home directory, including `.ssh/`) = ❌ **Lost forever**

So your SSH setup will be **gone** and you'll need to redo it.

## ✅ If You PAUSE/STOP the Pod

If RunPods has a pause/stop feature (keeps the instance):
- Everything persists, including `.ssh/config`
- You can resume and continue

## 🔄 Where Things Live

```bash
# Persistent (on FUSE network drive)
/workspace/ehr-sequencing/          ✅ Survives termination
/workspace/miniforge3/              ✅ Survives termination
/workspace/loinc-predictor/         ✅ Survives termination

# Ephemeral (on local pod storage)
/root/.ssh/                         ❌ Lost on termination
/root/.bashrc                       ❌ Lost on termination
/root/.gitconfig                    ❌ Lost on termination
```

## 💡 Solution: Save SSH Keys to /workspace

You can save your SSH setup to the persistent drive:

```bash
# Backup SSH config to workspace
cp -r ~/.ssh /workspace/.ssh-backup

# Next time on a new pod, restore:
cp -r /workspace/.ssh-backup/* ~/.ssh/
chmod 600 ~/.ssh/id_ed25519_github
```

Or create a setup script in `/workspace`:

```bash
# /workspace/setup-pod.sh
#!/bin/bash
# Restore SSH keys
cp /workspace/.ssh-backup/id_ed25519_github* ~/.ssh/
cp /workspace/.ssh-backup/config ~/.ssh/config
chmod 600 ~/.ssh/id_ed25519_github

# Restore git config
git config --global user.name "pleiadian53"
git config --global user.email "pleiadian53@users.noreply.github.com"

echo "✅ Pod setup restored!"
```

**For ephemeral pods (terminate after each session)**: You'll need to redo SSH setup each time, OR save keys to `/workspace` and restore them.