# Testing Documentation

This directory contains testing-related documentation and troubleshooting guides.

## Contents

- **[COMMON_ERRORS.md](COMMON_ERRORS.md)** - Common testing errors and their solutions
  - ModuleNotFoundError fixes
  - Environment activation issues
  - Import errors
  - Test collection problems
  - Dependency conflicts

## Purpose

This directory serves as a knowledge base for:
1. **Error Resolution** - Quick reference for fixing common test failures
2. **Best Practices** - Guidelines for writing and running tests
3. **Troubleshooting** - Step-by-step debugging procedures

## Quick Links

### Most Common Issues

1. **Package not installed** → See [Error 1](COMMON_ERRORS.md#error-1-modulenotfounderror-no-module-named-ehrsequencing)
2. **Wrong environment** → See [Error 3](COMMON_ERRORS.md#error-3-wrong-environment-activated)
3. **pytest not found** → See [Error 2](COMMON_ERRORS.md#error-2-pytest-command-not-found)

### Testing Guides

- [Testing Workflow](../workflow/TESTING_WORKFLOW.md) - Complete testing process
- [Test README](../../tests/README.md) - How to run tests

## Contributing

When you encounter a new testing error:

1. Document it in `COMMON_ERRORS.md`
2. Include:
   - Full error message
   - Root cause
   - Step-by-step solution
   - Prevention tips
3. Add to the index above

---

**Last Updated:** January 20, 2026
