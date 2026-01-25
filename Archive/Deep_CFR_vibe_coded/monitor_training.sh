#!/bin/bash
# Monitor Deep CFR training progress

echo "======================================================================"
echo "DEEP CFR TRAINING MONITOR"
echo "======================================================================"
echo ""

# Check if training is running
if ps aux | grep -v grep | grep "train.py" > /dev/null; then
    echo "✓ Training is RUNNING"
    ps aux | grep -v grep | grep "train.py" | awk '{print "  PID:", $2, "| CPU:", $3"%", "| Memory:", $4"%", "| Time:", $10}'
else
    echo "✗ Training is NOT running"
    echo "  Start with: uv run python Deep_CFR/train.py --epochs 100"
    exit 1
fi

echo ""
echo "----------------------------------------------------------------------"
echo "CHECKPOINTS"
echo "----------------------------------------------------------------------"

# Check checkpoints
if [ -d "../checkpoints" ]; then
    echo "Saved checkpoints:"
    ls -lht ../checkpoints/*.pt 2>/dev/null | head -5 | awk '{print "  " $9, "-", $6, $7, $8}'
    
    # Check latest checkpoint
    LATEST=$(ls -t ../checkpoints/deep_cfr_epoch*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        EPOCH=$(echo "$LATEST" | grep -o "epoch[0-9]*" | grep -o "[0-9]*")
        echo ""
        echo "Latest: Epoch $EPOCH of 100 ($(( EPOCH * 100 / 100 ))% complete)"
    fi
else
    echo "No checkpoints yet (training just started)"
fi

echo ""
echo "----------------------------------------------------------------------"
echo "ESTIMATED TIME"
echo "----------------------------------------------------------------------"

if [ -d "../checkpoints" ]; then
    LATEST=$(ls -t ../checkpoints/deep_cfr_epoch*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        EPOCH=$(echo "$LATEST" | grep -o "epoch[0-9]*" | grep -o "[0-9]*")
        REMAINING=$((100 - EPOCH))
        TIME_PER_EPOCH=240  # ~4 minutes per epoch
        REMAINING_SECONDS=$((REMAINING * TIME_PER_EPOCH))
        HOURS=$((REMAINING_SECONDS / 3600))
        MINUTES=$(( (REMAINING_SECONDS % 3600) / 60 ))
        echo "Epochs remaining: $REMAINING"
        echo "Estimated time: ${HOURS}h ${MINUTES}m"
    fi
else
    echo "Estimated total time: ~6-7 hours for 100 epochs"
    echo "Check back in 30 minutes to see progress"
fi

echo ""
echo "----------------------------------------------------------------------"
echo "COMMANDS"
echo "----------------------------------------------------------------------"
echo "Monitor:  watch -n 30 bash Deep_CFR/monitor_training.sh"
echo "Stop:     pkill -f train.py"
echo "Resume:   uv run python Deep_CFR/train.py --epochs 100"
echo ""

