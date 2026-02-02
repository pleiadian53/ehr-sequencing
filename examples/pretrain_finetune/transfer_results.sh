#!/bin/bash
# Transfer training results from pod to local system
# Usage: ./transfer_results.sh <pod-ip> <experiment-name>

set -e  # Exit on error

POD_IP=$1
EXPERIMENT_NAME=$2

if [ -z "$POD_IP" ] || [ -z "$EXPERIMENT_NAME" ]; then
    echo "Usage: ./transfer_results.sh <pod-ip> <experiment-name>"
    echo ""
    echo "Example:"
    echo "  ./transfer_results.sh 123.45.67.89 behrt_large_mlm_lora16"
    echo ""
    echo "This will transfer essential files from the pod:"
    echo "  - Plots (visualizations)"
    echo "  - Metrics history (logs/)"
    echo "  - Best checkpoint (checkpoints/best_lora_weights.pt)"
    echo "  - Hyperparameters and summaries (*.json, *.txt)"
    exit 1
fi

echo "=========================================="
echo "Transferring results from pod"
echo "=========================================="
echo "Pod IP: $POD_IP"
echo "Experiment: $EXPERIMENT_NAME"
echo ""

# Create local directory structure
echo "Creating local directories..."
mkdir -p experiments/$EXPERIMENT_NAME/plots
mkdir -p experiments/$EXPERIMENT_NAME/logs
mkdir -p experiments/$EXPERIMENT_NAME/checkpoints

# Transfer plots
echo ""
echo "📊 Transferring plots..."
scp -r root@$POD_IP:/workspace/ehr-sequencing/experiments/$EXPERIMENT_NAME/plots/* \
    experiments/$EXPERIMENT_NAME/plots/ 2>/dev/null || echo "  ⚠️  No plots found"

# Transfer logs
echo ""
echo "📝 Transferring logs..."
scp -r root@$POD_IP:/workspace/ehr-sequencing/experiments/$EXPERIMENT_NAME/logs/* \
    experiments/$EXPERIMENT_NAME/logs/ 2>/dev/null || echo "  ⚠️  No logs found"

# Transfer best checkpoint
echo ""
echo "💾 Transferring best checkpoint..."
scp root@$POD_IP:/workspace/ehr-sequencing/experiments/$EXPERIMENT_NAME/checkpoints/best_lora_weights.pt \
    experiments/$EXPERIMENT_NAME/checkpoints/ 2>/dev/null || echo "  ⚠️  No best checkpoint found"

# Transfer JSON files
echo ""
echo "📋 Transferring metadata..."
scp root@$POD_IP:/workspace/ehr-sequencing/experiments/$EXPERIMENT_NAME/*.json \
    experiments/$EXPERIMENT_NAME/ 2>/dev/null || echo "  ⚠️  No JSON files found"

# Transfer summary
echo ""
echo "📄 Transferring summary..."
scp root@$POD_IP:/workspace/ehr-sequencing/experiments/$EXPERIMENT_NAME/SUMMARY.txt \
    experiments/$EXPERIMENT_NAME/ 2>/dev/null || echo "  ⚠️  No summary found"

# Transfer training log if exists
echo ""
echo "📜 Transferring training log..."
scp root@$POD_IP:/workspace/ehr-sequencing/nohup.out \
    experiments/$EXPERIMENT_NAME/training.log 2>/dev/null || echo "  ⚠️  No training log found"

echo ""
echo "=========================================="
echo "✅ Transfer complete!"
echo "=========================================="
echo "Results saved to: experiments/$EXPERIMENT_NAME/"
echo ""
echo "Next steps:"
echo "  1. Verify transfer: ls -lh experiments/$EXPERIMENT_NAME/"
echo "  2. View summary: cat experiments/$EXPERIMENT_NAME/SUMMARY.txt"
echo "  3. Commit to Git: git add experiments/$EXPERIMENT_NAME/{plots,logs,*.json,*.txt}"
echo ""
