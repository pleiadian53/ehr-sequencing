# Synthea CSV Export Troubleshooting Guide

## Problem Statement

When generating synthetic patient data with Synthea, CSV files may not be exported even when `exporter.csv.export = true` is set in the `synthea.properties` configuration file. This document explains why this happens and provides reliable solutions.

## Symptoms

- Synthea runs successfully and reports generating patients (e.g., "You've just generated 1151 patients!")
- FHIR JSON files are created in `output/fhir/` directory
- CSV directory (`output/csv/`) is either empty or not created
- No error messages indicate CSV export failure

## Root Cause Analysis

### Why Configuration Files May Be Ignored

Synthea's configuration system has a specific precedence order:

1. **Command-line arguments** (highest priority)
2. **Local `synthea.properties` file** (in working directory)
3. **Embedded default properties** (in JAR file)

**Key Issue**: When running Synthea with `java -jar synthea-with-dependencies.jar`, the JAR may contain embedded default properties that override your local `synthea.properties` file, especially if:

- The JAR was built with specific export settings
- The properties file path is not correctly resolved
- The working directory doesn't match expectations

### What We Learned

Through multiple trials, we discovered:

1. **Properties files are not always read**: The `synthea.properties` file in the working directory may be ignored if the JAR has embedded defaults or if there's a path resolution issue.

2. **FHIR is often the default**: Many Synthea distributions default to FHIR export only, as FHIR is the modern healthcare data standard.

3. **Silent failures**: Synthea doesn't warn when it ignores export format settings - it simply exports in the default format.

4. **Command-line flags are reliable**: Using `--config*=value` flags guarantees the setting is applied, bypassing any configuration file issues.

## Solution: Command-Line Configuration Override

### Recommended Approach

Always use command-line flags to explicitly control export formats:

```bash
cd ~/work/synthea

# Generate 1000 patients with CSV export only
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  -p 1000
```

### Why This Works

- **Explicit control**: Command-line arguments have the highest precedence
- **No ambiguity**: Settings are visible in the command itself
- **Reproducible**: Anyone can see exactly what settings were used
- **Portable**: Works regardless of local configuration files

### Additional Export Control Options

```bash
# CSV only (recommended for EHR sequence modeling)
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  --exporter.ccda.export=false \
  --exporter.text.export=false \
  -p 1000

# Both CSV and FHIR (for interoperability testing)
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=true \
  -p 1000

# CSV with specific output directory
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  --exporter.baseDirectory=./custom_output/ \
  -p 1000
```

## Verification Steps

After running Synthea, verify CSV export was successful:

```bash
# Check if CSV directory exists and contains files
ls -lh ~/work/synthea/output/csv/

# Count patients in CSV file (should be N+1 for N patients due to header)
wc -l ~/work/synthea/output/csv/patients.csv

# Check file sizes (CSV files should be substantial, not empty)
du -sh ~/work/synthea/output/csv/
```

Expected output for 1,000 patients:
- `patients.csv`: ~300-400 KB
- `encounters.csv`: ~2-5 MB
- `observations.csv`: ~10-20 MB
- Total directory size: ~30-50 MB

## Troubleshooting Workflow

### Step 1: Verify Synthea Installation

```bash
cd ~/work/synthea
java -jar synthea-with-dependencies.jar --help
```

Should display help text without errors.

### Step 2: Test with Minimal Command

```bash
# Generate 10 patients with explicit CSV export
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  -p 10
```

### Step 3: Check Output

```bash
ls -la output/csv/
```

If CSV files appear, the issue was configuration precedence.

### Step 4: Scale Up

Once verified, generate full dataset:

```bash
# Clean previous output
rm -rf output/*

# Generate desired number of patients
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  -p 1000
```

## Common Pitfalls and Solutions

### Pitfall 1: Assuming Properties File Works

**Problem**: Editing `synthea.properties` but seeing no effect.

**Solution**: Use command-line flags instead of relying on properties file.

**Why**: JAR may have embedded defaults or path resolution issues.

### Pitfall 2: Not Verifying Output Format

**Problem**: Assuming CSV was generated without checking.

**Solution**: Always verify with `ls` and `wc -l` commands.

**Why**: Synthea doesn't fail or warn when exporting different format than expected.

### Pitfall 3: Reusing Old Output Directory

**Problem**: Mixing old and new data in same output directory.

**Solution**: Clear output directory before each generation:

```bash
rm -rf ~/work/synthea/output/*
```

**Why**: Synthea may append or skip existing files depending on configuration.

### Pitfall 4: Wrong Working Directory

**Problem**: Running Synthea from different directory than where JAR is located.

**Solution**: Always `cd` to Synthea directory first:

```bash
cd ~/work/synthea
java -jar synthea-with-dependencies.jar [options]
```

**Why**: Relative paths in configuration may break if working directory is wrong.

## Best Practices

### 1. Use Explicit Command-Line Flags

```bash
# Good: Explicit and clear
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  -p 1000

# Avoid: Relying on properties file
# (Just running with -p 1000 and hoping properties file works)
```

### 2. Document Your Generation Command

Save the exact command used in a script or README:

```bash
# generate_data.sh
#!/bin/bash
cd ~/work/synthea
rm -rf output/csv/*
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  -p 1000 \
  2>&1 | tee generation.log
```

### 3. Verify Before Moving Data

```bash
# Generate
java -jar synthea-with-dependencies.jar [options]

# Verify
echo "Checking CSV output..."
ls -lh output/csv/patients.csv
wc -l output/csv/patients.csv

# Only move if verification passes
if [ -f output/csv/patients.csv ]; then
    cp -r output/csv/* ~/work/loinc-predictor/data/synthea/large_cohort_1000/
    echo "Data copied successfully"
else
    echo "ERROR: CSV files not generated!"
    exit 1
fi
```

### 4. Keep Generation Logs

```bash
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  -p 1000 \
  2>&1 | tee synthea_generation_$(date +%Y%m%d_%H%M%S).log
```

This helps debug issues and provides a record of what was generated.

## Understanding Synthea's Export System

### Export Formats Available

- **CSV**: Tabular format, best for data analysis and ML pipelines
- **FHIR**: JSON format, healthcare interoperability standard
- **CCDA**: XML format, clinical document architecture
- **Text**: Human-readable clinical notes
- **CPCDS**: Claims data format

### Why CSV for EHR Sequence Modeling?

1. **Easy to parse**: Standard pandas/CSV libraries work out of the box
2. **Efficient**: Smaller file sizes than JSON
3. **Relational**: Natural fit for patient-visit-event hierarchy
4. **Familiar**: Most data scientists are comfortable with CSV
5. **Fast loading**: Faster than parsing nested JSON structures

### When to Use Other Formats

- **FHIR**: When testing FHIR-based pipelines or interoperability
- **Both CSV + FHIR**: When you need both analysis and standards compliance
- **Text**: When working with NLP models on clinical notes

## Summary of Lessons Learned

1. **Configuration precedence matters**: Command-line > local file > embedded defaults
2. **Explicit is better than implicit**: Always specify export format explicitly
3. **Verify, don't assume**: Check output before proceeding
4. **Document your process**: Save commands and logs for reproducibility
5. **Clean slate approach**: Clear output directory to avoid confusion
6. **Test small first**: Generate 10 patients before generating 1000

## Quick Reference

### Generate CSV Only (Most Common)

```bash
cd ~/work/synthea
rm -rf output/csv/*
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  -p 1000
```

### Verify Success

```bash
wc -l output/csv/patients.csv  # Should be 1001 (1000 + header)
du -sh output/csv/              # Should be ~30-50 MB
```

### Copy to Project

```bash
cp -r output/csv/* ~/work/loinc-predictor/data/synthea/large_cohort_1000/
```

## Related Documentation

- [Data Generation Guide](./data_generation_guide.md) - Overview of generating synthetic data
- [Synthea Documentation](https://github.com/synthetichealth/synthea/wiki) - Official Synthea wiki
- [Synthea Configuration](https://github.com/synthetichealth/synthea/wiki/Basic-Setup-and-Running#configuration) - Configuration system details

## Troubleshooting Checklist

- [ ] Synthea JAR file exists and is executable
- [ ] Running from correct directory (`cd ~/work/synthea`)
- [ ] Using command-line flags for export format
- [ ] Output directory is clean or cleared
- [ ] Verified CSV files are created after generation
- [ ] Checked file sizes are reasonable (not empty)
- [ ] Patient count matches expected (N+1 lines in patients.csv)
- [ ] Documented exact command used for reproducibility
