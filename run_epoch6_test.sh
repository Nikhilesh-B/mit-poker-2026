#!/bin/bash
# Quick test of Epoch 6 model vs Henry

cd /Users/nikhileshbelulkar/Documents/mit-poker-2026

echo "======================================================================"
echo "TESTING EPOCH 6 vs HENRY (100 rounds)"
echo "======================================================================"
echo ""
echo "✓ Epoch 6 checkpoint loaded"
echo "✓ Config set to 100 rounds (quick test)"
echo ""
echo "Starting match..."
echo ""

# Run the engine (use the method that works for your setup)
python3 engine.py

echo ""
echo "======================================================================"
echo "MATCH COMPLETE!"
echo "======================================================================"
echo ""
echo "Check results:"
echo "  • Latest gamelog in gamelogs/ folder"
echo "  • Look for final score at end of game"
echo ""

