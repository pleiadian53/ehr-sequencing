# Pod Management Utilities

Scripts for managing cloud GPU pods (RunPods, Lambda Labs, etc.) for the ehr-sequencing project.

## Overview

These utilities help you:
- Configure SSH access to pods with memorable hostnames
- Transfer training results efficiently
- Manage multiple pods across different projects
- Maintain SSH config history and backups

## Scripts

### 1. `runpod_ssh_manager.sh`

Manage SSH configurations for RunPods instances.

**Features:**
- Add/update/remove pod configurations
- Automatic SSH config backup before changes
- History tracking of all pods
- Interactive menu or command-line interface
- Test connections after setup

**Usage:**

```bash
# Interactive menu
./runpod_ssh_manager.sh

# Add a new pod
./runpod_ssh_manager.sh add ehr-sequencing

# List current pods
./runpod_ssh_manager.sh list

# Show history
./runpod_ssh_manager.sh history

# Remove a pod
./runpod_ssh_manager.sh remove
```

**Adding a Pod (Interactive):**

```bash
./runpod_ssh_manager.sh add ehr-sequencing

# You'll be prompted for:
# - Pod Hostname/IP: 69.30.85.45
# - Pod Port: 22177
# - Pod Nickname: a40-main
# - SSH Key Path: [~/.ssh/id_ed25519]

# This creates an SSH config entry:
# Host runpod-ehr-sequencing-a40-main
#     HostName 69.30.85.45
#     Port 22177
#     User root
#     IdentityFile ~/.ssh/id_ed25519
#     StrictHostKeyChecking no
#     UserKnownHostsFile /dev/null
#     ServerAliveInterval 60
#     ServerAliveCountMax 5
#     ConnectTimeout 10
#     Compression yes
```

**Now you can use the hostname:**

```bash
# SSH into pod
ssh runpod-ehr-sequencing-a40-main

# Transfer files
scp file.txt runpod-ehr-sequencing-a40-main:/workspace/

# Use with transfer_results.sh
cd examples/pretrain_finetune
./transfer_results.sh runpod-ehr-sequencing-a40-main behrt_large_mlm_lora16
```

### 2. `transfer_results.sh` (in examples/pretrain_finetune/)

Transfer training results from pod to local system.

**Features:**
- Accepts SSH hostnames or raw IPs
- Transfers only essential files (plots, metrics, best checkpoint)
- Automatic directory creation
- Clear progress messages

**Usage:**

```bash
cd examples/pretrain_finetune

# Using SSH hostname (recommended)
./transfer_results.sh runpod-ehr-sequencing-a40-main behrt_large_mlm_lora16

# Using raw IP
./transfer_results.sh 123.45.67.89 behrt_large_mlm_lora16
```

## Quick Start Workflow

### 1. Set Up Pod Access

```bash
# Add your pod to SSH config
cd scripts/pod_management
./runpod_ssh_manager.sh add ehr-sequencing

# Follow prompts to enter pod details
# Test connection
ssh runpod-ehr-sequencing-a40-main
```

### 2. Train on Pod

```bash
# SSH into pod
ssh runpod-ehr-sequencing-a40-main

# Clone repo and setup (first time only)
git clone https://github.com/yourusername/ehr-sequencing.git
cd ehr-sequencing
pip install -e .

# Run training (auto-detects A40 and sets optimal params!)
python examples/pretrain_finetune/train_behrt_demo.py --demo_data

# Or run in background
nohup python examples/pretrain_finetune/train_behrt_demo.py --demo_data > training.log 2>&1 &
```

### 3. Transfer Results

```bash
# On local machine
cd examples/pretrain_finetune
./transfer_results.sh runpod-ehr-sequencing-a40-main behrt_large_mlm_lora16

# Verify transfer
ls -lh experiments/behrt_large_mlm_lora16/
cat experiments/behrt_large_mlm_lora16/SUMMARY.txt
```

### 4. Commit Results

```bash
# Commit plots and summaries (checkpoints are in .gitignore)
git add experiments/behrt_large_mlm_lora16/{plots,logs,*.json,*.txt}
git commit -m "Add training results: BEHRT large LoRA"
git push
```

## SSH Config Management

### Backup and Restore

The `runpod_ssh_manager.sh` automatically backs up your SSH config before making changes.

```bash
# List backups
./runpod_ssh_manager.sh backups

# Restore from backup
./runpod_ssh_manager.sh restore
```

Backups are stored in `~/.ssh/config_backups/`

### History Tracking

All pod configurations are tracked in `~/.ssh/runpod_history.json`

```bash
# View history
./runpod_ssh_manager.sh history

# Shows:
# - Project name
# - Host alias
# - When added
# - Status
```

## Tips and Best Practices

### SSH Hostname Naming

The script creates hostnames in the format:
```
runpod-{project}-{nickname}
```

Examples:
- `runpod-ehr-sequencing-a40-main`
- `runpod-ehr-sequencing-v100-backup`
- `runpod-genai-lab-a100`

**Tip:** Use descriptive nicknames like `a40-main`, `v100-dev`, `a100-prod`

### Multiple Pods

You can manage multiple pods for the same project:

```bash
# Main training pod
./runpod_ssh_manager.sh add ehr-sequencing
# Nickname: a40-main

# Backup pod
./runpod_ssh_manager.sh add ehr-sequencing
# Nickname: v100-backup

# Development pod
./runpod_ssh_manager.sh add ehr-sequencing
# Nickname: t4-dev
```

Now you have:
- `ssh runpod-ehr-sequencing-a40-main`
- `ssh runpod-ehr-sequencing-v100-backup`
- `ssh runpod-ehr-sequencing-t4-dev`

### Updating Pod Details

If your pod IP or port changes:

```bash
# Remove old entry
./runpod_ssh_manager.sh remove
# Select the pod to remove

# Add new entry with same nickname
./runpod_ssh_manager.sh add ehr-sequencing
# Use the same nickname
```

Or use the update option:

```bash
./runpod_ssh_manager.sh add ehr-sequencing
# Enter same nickname
# Confirm update when prompted
```

### Connection Issues

If you can't connect:

```bash
# Test connection manually
ssh -v runpod-ehr-sequencing-a40-main

# Check SSH config
cat ~/.ssh/config | grep -A 10 "runpod-ehr-sequencing"

# Verify pod is running in RunPods dashboard
# Check IP and port haven't changed
```

## Integration with Training Scripts

The training scripts in `examples/pretrain_finetune/` are designed to work seamlessly with these pod management utilities:

### Auto Resource Detection

```bash
# Just works on any pod!
python train_behrt_demo.py --demo_data
```

The script automatically:
- Detects GPU type (A40, A100, V100, etc.)
- Sets optimal batch size, model size, epochs
- Configures LoRA rank appropriately

### Transfer Results

```bash
# Use the SSH hostname you configured
./transfer_results.sh runpod-ehr-sequencing-a40-main behrt_large_mlm_lora16
```

The transfer script:
- Recognizes SSH hostnames from config
- Uses proper port, compression, timeout settings
- Transfers only essential files

## Directory Structure

```
scripts/pod_management/
├── README.md                    # This file
└── runpod_ssh_manager.sh       # SSH config manager

examples/pretrain_finetune/
├── transfer_results.sh         # Result transfer utility
├── train_behrt_demo.py        # Training script with auto resource detection
└── POD_WORKFLOW.md            # Complete pod workflow guide

~/.ssh/
├── config                      # SSH config (managed by script)
├── config_backups/            # Automatic backups
└── runpod_history.json        # Pod history tracking
```

## Troubleshooting

### Script Permission Denied

```bash
chmod +x scripts/pod_management/runpod_ssh_manager.sh
chmod +x examples/pretrain_finetune/transfer_results.sh
```

### Python3 Not Found (for history tracking)

The history feature requires Python 3. If not available:
- History tracking will be skipped (warning shown)
- All other features work normally
- Install Python 3 to enable history

### SSH Config Conflicts

If you have existing SSH config entries:
- Script backs up before making changes
- You can restore from backup if needed
- Manual entries are preserved

### Transfer Fails

```bash
# Check pod is accessible
ssh runpod-ehr-sequencing-a40-main "echo 'Connected'"

# Check experiment exists on pod
ssh runpod-ehr-sequencing-a40-main "ls -la /workspace/ehr-sequencing/experiments/"

# Try manual transfer to debug
scp runpod-ehr-sequencing-a40-main:/workspace/ehr-sequencing/experiments/behrt_large_mlm_lora16/SUMMARY.txt .
```

## See Also

- `examples/pretrain_finetune/POD_WORKFLOW.md` - Complete pod training workflow
- `examples/pretrain_finetune/README.md` - Training scripts documentation
- `src/ehrsequencing/utils/resource_manager.py` - Auto resource detection

## Contributing

To add new pod management utilities:

1. Add script to `scripts/pod_management/`
2. Make it executable: `chmod +x script.sh`
3. Update this README
4. Update `POD_WORKFLOW.md` if relevant
5. Test on actual pod before committing
