#!/bin/bash
# Quick script to test a specific checkpoint

if [ -z "$1" ]; then
    echo "Usage: bash test_checkpoint.sh <epoch_number>"
    echo ""
    echo "Available checkpoints:"
    ls -1 ../checkpoints/deep_cfr_epoch*.pt 2>/dev/null | sed 's/.*epoch/  Epoch /' | sed 's/.pt//'
    echo ""
    echo "Example: bash test_checkpoint.sh 10"
    exit 1
fi

EPOCH=$1
CHECKPOINT="../checkpoints/deep_cfr_epoch${EPOCH}.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -1 ../checkpoints/deep_cfr_epoch*.pt 2>/dev/null | sed 's/.*epoch/  Epoch /' | sed 's/.pt//'
    exit 1
fi

echo "======================================================================"
echo "TESTING CHECKPOINT: Epoch $EPOCH"
echo "======================================================================"
echo ""

# Temporarily swap checkpoint to test it
cp "$CHECKPOINT" "../checkpoints/deep_cfr_final.pt"
echo "✓ Loaded checkpoint from epoch $EPOCH"
echo ""

# Run test
uv run python test_trained_model.py

echo ""
echo "======================================================================"
echo "To play against this version:"
echo "  Terminal 1: python ../engine.py"
echo "  Terminal 2: cd Deep_CFR && uv run python player.py"
echo "======================================================================"

