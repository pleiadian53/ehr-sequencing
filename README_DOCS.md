# EHR Sequencing Documentation Guide

Complete guide for writing and deploying documentation with MathJax support.

---

## Quick Start

### Install Dependencies

Documentation dependencies are included in the environment files. If you already have the environment set up:

```bash
# Option 1: Update existing environment
mamba activate ehrsequencing
mamba env update -f environment-macos.yml  # or environment-cuda.yml / environment-cpu.yml

# Option 2: Install just the docs dependencies
mamba activate ehrsequencing
pip install mkdocs mkdocs-material pymdown-extensions mkdocs-jupyter
```

For new installations, the documentation tools are automatically included when you create the environment.

### Build Locally
```bash
mkdocs serve  # Opens at http://127.0.0.1:8000/
```

### Deploy to GitHub Pages

#### First-Time Setup (One-Time Only)

**Step 1: Manual deployment to create gh-pages branch**
```bash
# Build and push to gh-pages branch
mkdocs gh-deploy

# This creates the gh-pages branch and deploys the site
```

**Step 2: Configure GitHub Pages**
1. Go to: `https://github.com/pleiadian53/ehr-sequencing/settings/pages`
2. **Source:** Select "Deploy from a branch"
3. **Branch:** Select `gh-pages`
4. **Folder:** Select `/ (root)`
5. Click "Save"

**Step 3: Wait 5-10 minutes**

Your site will be live at: `https://pleiadian53.github.io/ehr-sequencing/`

#### Automatic Deployment (After Setup)

Once configured, documentation auto-deploys on every push to `main`:

```bash
# 1. Make changes to documentation
vim docs/your-file.md

# 2. Commit and push
git add docs/your-file.md
git commit -m "Update documentation"
git push origin main

# 3. GitHub Actions automatically:
#    - Builds site
#    - Deploys to gh-pages
#    - Updates live site (2-3 minutes)
```

**Monitor deployment:** https://github.com/pleiadian53/ehr-sequencing/actions

---

## Writing Math

**Inline**: `$h_t = P(T = t \mid T \geq t)$` → $h_t = P(T = t \mid T \geq t)$

**Display**:
```markdown
$$
S(t) = \prod_{i=1}^{t} (1 - h_i)
$$
```

---

## Adding Notebooks

Place `.ipynb` files in `notebooks/` directory. They'll auto-render with:
- Code cells with syntax highlighting
- Output cells (plots, tables)
- Markdown cells with LaTeX support

---

## Configuration Files

1. **`mkdocs.yml`** - Main config (navigation, theme, plugins)
2. **`requirements-docs.txt`** - Python dependencies
3. **`.github/workflows/docs.yml`** - Auto-deployment
4. **`docs/javascripts/mathjax.js`** - MathJax config
5. **`docs/stylesheets/extra.css`** - Custom styles

---

## Navigation

Edit `mkdocs.yml` to add pages:
```yaml
nav:
  - Home: index.md
  - 'Survival Analysis':
      - 'Tutorial 1': notebooks/02_survival_analysis/TUTORIAL_01_prediction_problem.md
      - 'Notebook': notebooks/02_survival_analysis/01_discrete_time_survival_lstm.ipynb
```

---

## Troubleshooting

**Math not rendering**: Check browser console for MathJax errors

**Notebook not showing**: Verify `mkdocs-jupyter` plugin installed

**Build fails**: Run `mkdocs build --verbose` for details

---

## Features

✅ LaTeX math with MathJax  
✅ Jupyter notebook rendering  
✅ Dark mode support  
✅ Mobile responsive  
✅ Search functionality  
✅ Auto-deployment to GitHub Pages

---

**Documentation URL**: https://pleiadian53.github.io/ehr-sequencing/
