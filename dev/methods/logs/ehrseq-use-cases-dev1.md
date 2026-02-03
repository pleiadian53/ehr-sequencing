```
# Create sessions directory at project root
mkdir -p /workspace/ehr-sequencing/experiments/sessions

# Use absolute paths in nohup
cd /workspace/ehr-sequencing/examples/pretrain_finetune
nohup python -u benchmark_pretrained_embeddings.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --batch-size 128 \
    --output-dir /workspace/ehr-sequencing/experiments/benchmark_embeddings \
    > /workspace/ehr-sequencing/experiments/sessions/embeddings_comparison_large.out 2>&1 &
```

---

```
/workspace/ehr-sequencing/
├── experiments/                    # All experiment outputs
│   ├── sessions/                   # nohup logs
│   │   ├── embeddings_comparison_large.out
│   │   ├── behrt_vs_pyhealth_large.out
│   │   └── ...
│   ├── benchmark_embeddings/       # Benchmark results
│   │   ├── SUMMARY.txt
│   │   ├── training_curves.png
│   │   └── ...
│   └── behrt_vs_pyhealth/         # Comparison results
│       └── ...
```

---

