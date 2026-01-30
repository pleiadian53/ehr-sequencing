# MkDocs Troubleshooting (2026-01-25)

## CRITICAL: Symlinks Don't Work ⚠️

**Problem:** Notebooks symlinked but links return 404

**Solution:**
```bash
cd docs/
rm notebooks INSTALL.md  # Remove symlinks
cp -r ../notebooks notebooks  # Copy files
cp ../INSTALL.md INSTALL.md
```

**Why:** MkDocs can't resolve links in symlinked files

**Reference:** https://stackoverflow.com/questions/77897139/building-documentation-behind-a-symlink

---

## Other Common Issues

### 1. Badges Not Rendering
- **Problem:** `<div align="center">` wrapper breaks badges
- **Solution:** Remove wrapper, use plain markdown

### 2. Missing Mermaid Diagrams
- **Problem:** Diagrams not in docs/index.md
- **Solution:** Copy from README.md to docs/index.md

### 3. Broken dev/ Links
- **Problem:** Links to private dev/ folder
- **Solution:** Remove all dev/ references from public docs

### 4. Wrong Relative Paths
- **Problem:** Paths like `../../docs/methods/` from within docs/
- **Solution:** Use `methods/` not `../../docs/methods/`

### 5. Notebooks Not in Navigation
- **Problem:** Files copied but not visible in menu
- **Solution:** Add to mkdocs.yml nav section

---

## Checklist for New Projects

- [ ] Copy files to docs/ (don't symlink)
- [ ] Add all sections to mkdocs.yml nav
- [ ] Remove dev/ references
- [ ] Fix relative paths
- [ ] Remove HTML wrappers from badges
- [ ] Test locally: `mkdocs serve`
- [ ] Verify all links work
