#!/bin/bash
# Quick script to check training progress

echo "======================================================================"
echo "DEEP CFR TRAINING PROGRESS CHECK"
echo "======================================================================"
echo ""

# Check if training is running
if ps aux | grep "train.py" | grep -v grep > /dev/null; then
    echo "✅ Training is RUNNING"
    ps aux | grep "train.py" | grep -v grep | awk '{print "   PID: " $2 ", CPU: " $3 "%, Runtime: " $10}'
    echo ""
else
    echo "❌ Training is NOT running"
    echo ""
fi

# Show latest log output
echo "📋 Latest log output:"
echo "----------------------------------------------------------------------"
tail -30 Deep_CFR/training_100epoch.log
echo "----------------------------------------------------------------------"
echo ""

# Check for checkpoints
echo "💾 Checkpoints created:"
ls -lht ../checkpoints/deep_cfr_epoch*.pt 2>/dev/null | head -5 | awk '{print "   " $9 " - " $6 " " $7 " " $8}'
echo ""

# Check training metrics
if [ -f "Deep_CFR/logs/training_metrics.csv" ]; then
    echo "📊 Latest metrics:"
    tail -3 Deep_CFR/logs/training_metrics.csv | column -t -s,
    echo ""
fi

# Check evaluation results
if [ -f "Deep_CFR/logs/evaluation_results.csv" ]; then
    echo "🎮 Latest evaluation results:"
    tail -3 Deep_CFR/logs/evaluation_results.csv | column -t -s,
    echo ""
fi

echo "======================================================================"
echo "To monitor live: tail -f Deep_CFR/training_100epoch.log"
echo "To kill training: pkill -f 'python.*train.py'"
echo "To generate plots: uv run python Deep_CFR/plot_training.py"
echo "======================================================================"
