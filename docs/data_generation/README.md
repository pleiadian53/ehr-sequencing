# Data Generation Documentation

This directory contains comprehensive guides for generating synthetic patient data using Synthea for EHR sequence modeling and survival analysis.

## Documents

### 1. [Data Generation Guide](./data_generation_guide.md)
**Purpose**: Main guide for generating synthetic patient data with Synthea

**Topics covered**:
- Why synthetic data is needed for EHR deep learning
- Dataset size recommendations for different use cases
- Installing and setting up Synthea
- Generating different patient cohorts
- Configuring Synthea for specific conditions
- Integrating generated data into training pipelines
- Expected model performance by dataset size

**When to use**: Start here for overview and general guidance on data generation.

### 2. [Synthea CSV Export Troubleshooting](./synthea_csv_export_troubleshooting.md)
**Purpose**: Detailed troubleshooting guide for CSV export issues

**Topics covered**:
- Why CSV files may not be generated despite configuration
- Root cause analysis of configuration precedence
- Reliable solutions using command-line flags
- Verification steps and best practices
- Common pitfalls and how to avoid them
- Lessons learned from trial-and-error debugging

**When to use**: Reference this when CSV files are not being generated, or to understand the correct way to ensure CSV export.

## Quick Start

### Generate 1000 Patients (CSV Format)

```bash
cd ~/work/synthea

# Clean output directory
rm -rf output/csv/*

# Generate with explicit CSV export
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  -p 1000

# Verify success
wc -l output/csv/patients.csv  # Should show 1001 (1000 + header)
du -sh output/csv/              # Should be ~30-50 MB

# Copy to project
cp -r output/csv/* ~/work/loinc-predictor/data/synthea/large_cohort_1000/
```

## Key Lessons

### Configuration Precedence
Command-line arguments > Local properties file > Embedded JAR defaults

**Always use command-line flags** to ensure settings are applied.

### Verification is Critical
Never assume data was generated in the expected format. Always verify:
- CSV directory exists
- Files have reasonable sizes
- Patient count matches expectation

### Document Your Process
Save the exact commands used for reproducibility and debugging.

## Common Use Cases

### Small Dataset for Testing (100 patients)
```bash
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  -p 100
```

### Medium Dataset for Development (1000 patients)
```bash
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  -p 1000
```

### Large Dataset for Training (10000 patients)
```bash
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  -p 10000
```

### Disease-Specific Cohort (CKD patients)
```bash
java -jar synthea-with-dependencies.jar \
  --exporter.csv.export=true \
  --exporter.fhir.export=false \
  -p 1000 \
  -m chronic_kidney_disease
```

## Related Documentation

- [Pretrained Embeddings Guide](../pretrained_embeddings_guide.md) - Using pretrained medical code embeddings
- [Survival Analysis Methods](../methods/causal-survival-analysis-1.md) - Causal survival analysis theory
- [Synthea Official Wiki](https://github.com/synthetichealth/synthea/wiki) - Comprehensive Synthea documentation

## Troubleshooting Decision Tree

```
CSV files not generated?
│
├─ Are FHIR files being created instead?
│  └─ YES → See "Synthea CSV Export Troubleshooting" guide
│
├─ Is Synthea running without errors?
│  ├─ NO → Check Java installation and JAR file
│  └─ YES → Use command-line flags instead of properties file
│
└─ Are you in the correct directory?
   └─ Run: cd ~/work/synthea
```

## Best Practices Summary

1. **Use explicit command-line flags** for export format control
2. **Clean output directory** before each generation
3. **Verify output** immediately after generation
4. **Document commands** in scripts or logs
5. **Test with small datasets** before scaling up
6. **Check file sizes** to ensure data was actually generated

## Support

If you encounter issues not covered in these guides:

1. Check the [Synthea GitHub Issues](https://github.com/synthetichealth/synthea/issues)
2. Review the [Synthea Wiki](https://github.com/synthetichealth/synthea/wiki)
3. Verify your Java version: `java -version` (requires Java 11+)
4. Check Synthea version and update if needed

## Contributing

When adding new data generation documentation:

- Place files in this directory
- Update this README with links and descriptions
- Include practical examples and code snippets
- Document lessons learned from troubleshooting
- Add to the "Common Use Cases" section if applicable
