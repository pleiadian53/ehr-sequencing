# IDE and Jupyter Notebook Setup

Guide for configuring VS Code, Cursor, and Jupyter notebooks to use your `ehrsequencing` conda environment.

---

## 🎯 Overview

After creating the conda environment, you need to:
1. Register the environment as a Jupyter kernel (for notebooks)
2. Configure your IDE to recognize the Python interpreter
3. Select the correct kernel/interpreter in your workspace

**Estimated time**: 2-3 minutes

---

## ⚡ Quick Setup (Copy-Paste)

```bash
# 1. Install Jupyter kernel for the environment
mamba run -n ehrsequencing python -m ipykernel install --user --name=ehrsequencing --display-name="EHR Sequencing (Python 3.10)"

# 2. Verify kernel is installed
jupyter kernelspec list

# 3. Get the Python interpreter path (for IDE)
echo "/workspace/miniforge3/envs/ehrsequencing/bin/python"
```

After running these commands:
- **For Jupyter notebooks**: Select "EHR Sequencing (Python 3.10)" as the kernel
- **For Python scripts**: Use interpreter path: `/workspace/miniforge3/envs/ehrsequencing/bin/python`

---

## 📋 Step-by-Step Instructions

### Step 1: Install Jupyter Kernel

The Jupyter kernel makes your conda environment available to Jupyter notebooks.

```bash
# Option A: If mamba activate works
eval "$(mamba shell hook --shell bash)"
mamba activate ehrsequencing
python -m ipykernel install --user --name=ehrsequencing --display-name="EHR Sequencing (Python 3.10)"

# Option B: If mamba activate doesn't work, use mamba run
mamba run -n ehrsequencing python -m ipykernel install --user --name=ehrsequencing --display-name="EHR Sequencing (Python 3.10)"
```

**What this does**:
- Registers the environment's Python interpreter as a Jupyter kernel
- Name: `ehrsequencing` (internal identifier)
- Display name: "EHR Sequencing (Python 3.10)" (what you see in UI)
- Location: `~/.local/share/jupyter/kernels/ehrsequencing/`

**Verify installation**:
```bash
jupyter kernelspec list
```

**Expected output**:
```
Available kernels:
  ehrsequencing    /root/.local/share/jupyter/kernels/ehrsequencing
  python3          /usr/local/share/jupyter/kernels/python3
```

✅ **Jupyter kernel installed!**

---

### Step 2: Configure IDE Python Interpreter

Your IDE (VS Code, Cursor, PyCharm) needs to know where the Python interpreter is located.

#### Find Your Interpreter Path

```bash
# Display the full path to the Python interpreter
echo "/workspace/miniforge3/envs/ehrsequencing/bin/python"

# Or verify it exists
ls -la /workspace/miniforge3/envs/ehrsequencing/bin/python
```

**Path to use**:
```
/workspace/miniforge3/envs/ehrsequencing/bin/python
```

#### VS Code / Cursor

**Method 1: Command Palette (Recommended)**

1. Open Command Palette:
   - **Windows/Linux**: `Ctrl + Shift + P`
   - **macOS**: `Cmd + Shift + P`

2. Type: `Python: Select Interpreter`

3. If you see "EHR Sequencing" or "ehrsequencing" in the list:
   - ✅ Select it
   - Done!

4. If you **don't** see it:
   - Click **"Enter interpreter path..."**
   - Click **"Find..."**
   - Navigate to: `/workspace/miniforge3/envs/ehrsequencing/bin/`
   - Select `python` or `python3.10`
   
   **OR** manually enter:
   ```
   /workspace/miniforge3/envs/ehrsequencing/bin/python
   ```

**Method 2: Status Bar (Quick Access)**

1. Look at the **bottom-right** corner of your IDE
2. You'll see the current Python version (e.g., "Python 3.12.12")
3. **Click** on it
4. Select "EHR Sequencing" or enter the interpreter path

**Method 3: Settings (Workspace-Specific)**

Create or edit `.vscode/settings.json` in your project:

```json
{
    "python.defaultInterpreterPath": "/workspace/miniforge3/envs/ehrsequencing/bin/python",
    "jupyter.kernels.filter": [
        {
            "path": "/workspace/miniforge3/envs/ehrsequencing/bin/python",
            "type": "pythonEnvironment"
        }
    ]
}
```

#### PyCharm

1. **File** → **Settings** (or `Ctrl + Alt + S`)
2. **Project: ehr-sequencing** → **Python Interpreter**
3. Click the **gear icon** ⚙️ → **Add...**
4. Select **Conda Environment** → **Existing environment**
5. Set interpreter path:
   ```
   /workspace/miniforge3/envs/ehrsequencing/bin/python
   ```
6. Click **OK**

---

### Step 3: Select Kernel in Jupyter Notebooks

#### In Jupyter Notebook Interface (Browser)

1. Open your notebook
2. Click **Kernel** → **Change Kernel**
3. Select **"EHR Sequencing (Python 3.10)"**

#### In VS Code / Cursor (Integrated Notebooks)

1. Open your `.ipynb` notebook file
2. Look at the **top-right corner** of the notebook
3. Click on the **kernel selector** (shows current kernel/Python version)
4. Select **"EHR Sequencing (Python 3.10)"** from the dropdown

**If you don't see it**:
- Click **"Select Another Kernel..."**
- Choose **"Jupyter Kernel..."**
- Look for **"ehrsequencing"** or **"EHR Sequencing (Python 3.10)"**

#### In JupyterLab

1. Open your notebook
2. Click the **kernel name** in the top-right (e.g., "Python 3")
3. Select **"EHR Sequencing (Python 3.10)"** from the list

---

## ✅ Verify Setup

### Test 1: Check Python Interpreter

In a notebook cell or Python file, run:

```python
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
```

**Expected output**:
```
Python executable: /workspace/miniforge3/envs/ehrsequencing/bin/python
Python version: 3.10.x ...
```

### Test 2: Check Package Installation

```python
# Test EHR sequencing package
import ehrsequencing
print(f"EHR Sequencing installed: {ehrsequencing.__file__}")

# Test key dependencies
import torch
import pandas as pd
import lifelines
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

**If imports fail**, the wrong kernel is selected. Go back to Step 3.

### Test 3: List Available Kernels

In terminal:
```bash
jupyter kernelspec list
```

Should show:
```
ehrsequencing    /root/.local/share/jupyter/kernels/ehrsequencing
```

---

## 🔧 Troubleshooting

### Problem: Kernel not showing in notebook

**Symptom**: Can't find "EHR Sequencing" in kernel list

**Solutions**:

1. **Verify kernel is installed**:
   ```bash
   jupyter kernelspec list
   # Should show 'ehrsequencing'
   ```

2. **Restart your IDE completely**:
   - Close all notebook files
   - Exit IDE
   - Reopen IDE
   - Open notebook again

3. **Reinstall the kernel**:
   ```bash
   # Remove old kernel
   jupyter kernelspec remove ehrsequencing
   
   # Reinstall
   mamba run -n ehrsequencing python -m ipykernel install --user --name=ehrsequencing --display-name="EHR Sequencing (Python 3.10)"
   ```

4. **Check kernel.json**:
   ```bash
   cat ~/.local/share/jupyter/kernels/ehrsequencing/kernel.json
   ```
   
   Should show:
   ```json
   {
     "argv": [
       "/workspace/miniforge3/envs/ehrsequencing/bin/python",
       "-m",
       "ipykernel_launcher",
       "-f",
       "{connection_file}"
     ],
     "display_name": "EHR Sequencing (Python 3.10)",
     "language": "python"
   }
   ```

### Problem: IDE doesn't show the interpreter

**Symptom**: Interpreter path not visible in Python interpreter list

**Solutions**:

1. **Click the refresh button** in the interpreter selection dialog (↻ icon)

2. **Manually enter the path**:
   ```
   /workspace/miniforge3/envs/ehrsequencing/bin/python
   ```

3. **Check the environment exists**:
   ```bash
   ls -la /workspace/miniforge3/envs/ehrsequencing/bin/python
   # Should show the Python executable
   ```

4. **Reload IDE window**:
   - Command Palette → "Developer: Reload Window"
   - Or restart IDE completely

### Problem: Wrong packages imported

**Symptom**: Imports work but wrong versions, or `ehrsequencing` not found

**Cause**: Using the wrong Python interpreter/kernel

**Diagnosis**:
```python
import sys
print(sys.executable)
# Should show: /workspace/miniforge3/envs/ehrsequencing/bin/python
# If it shows something else, wrong interpreter is active
```

**Solution**: Go back to Step 2 or Step 3 and select the correct interpreter/kernel

### Problem: Kernel keeps dying

**Symptom**: "Kernel died" or "Kernel restarting" errors

**Common causes**:

1. **Out of memory**:
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # Check RAM
   free -h
   ```

2. **Package conflicts**:
   ```bash
   # Verify packages installed correctly
   mamba run -n ehrsequencing pip list
   ```

3. **Corrupted kernel**:
   ```bash
   # Reinstall kernel
   jupyter kernelspec remove ehrsequencing
   mamba run -n ehrsequencing python -m ipykernel install --user --name=ehrsequencing --display-name="EHR Sequencing (Python 3.10)"
   ```

### Problem: ipykernel not found

**Error**: `No module named 'ipykernel'`

**Solution**: Install ipykernel in the environment:
```bash
mamba run -n ehrsequencing pip install ipykernel
```

---

## 🚀 Advanced: Multiple Kernels

If you're working with multiple projects on the same pod:

### Install Additional Kernels

```bash
# For meta-spliceai project
mamba run -n metaspliceai python -m ipykernel install --user --name=metaspliceai --display-name="Meta SpliceAI (Python 3.11)"

# For agentic-spliceai project
mamba run -n agenticspliceai python -m ipykernel install --user --name=agenticspliceai --display-name="Agentic SpliceAI (Python 3.11)"
```

### List All Kernels

```bash
jupyter kernelspec list
```

### Remove Old/Unused Kernels

```bash
# List kernels
jupyter kernelspec list

# Remove specific kernel
jupyter kernelspec remove old-kernel-name
```

### Switch Kernels Mid-Notebook

You can change kernels without restarting:
1. Click kernel selector
2. Choose different kernel
3. Notebook cells will use new environment

**Note**: Variables/state are lost when switching kernels.

---

## 📊 Kernel Selection Best Practices

### For Notebooks

✅ **Do**:
- Select kernel **before** running cells
- Verify correct kernel with `sys.executable`
- Restart kernel after changing environments

❌ **Don't**:
- Mix packages from different environments
- Assume kernel matches your terminal environment
- Run notebooks with "Python 3" default kernel (usually wrong)

### For Python Scripts

✅ **Do**:
- Set interpreter at workspace level (`.vscode/settings.json`)
- Verify interpreter in status bar before running
- Use `mamba activate` in terminals

❌ **Don't**:
- Rely on system Python
- Mix conda and system pip installs
- Use global Python for project-specific code

---

## 📝 Summary Checklist

After setup, verify:

- [ ] Jupyter kernel installed (`jupyter kernelspec list` shows `ehrsequencing`)
- [ ] IDE recognizes interpreter (status bar shows "EHR Sequencing" or Python 3.10)
- [ ] Notebook kernel selector shows "EHR Sequencing (Python 3.10)"
- [ ] `sys.executable` points to `/workspace/miniforge3/envs/ehrsequencing/bin/python`
- [ ] `import ehrsequencing` works in notebook/script
- [ ] `torch.cuda.is_available()` returns `True` (if using GPU)

**Time to complete**: 2-3 minutes  
**Frequency**: Once per project/pod  
**Result**: IDE and Jupyter configured for EHR sequence modeling 🚀

---

## 🔗 Related Documentation

- **Environment Setup**: `POD_ENVIRONMENT_SETUP.md` - Creating conda environment
- **Code Sync**: `RSYNC_QUICK_REFERENCE.md` - Syncing code to pod
- **Local Development**: `LOCAL_DEVELOPMENT_WORKFLOW.md` - Development workflow

---

**Created**: January 29, 2026  
**Version**: 1.0.0  
**Applies to**: VS Code, Cursor, PyCharm, JupyterLab on RunPods
