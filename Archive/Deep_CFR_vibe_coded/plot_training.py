#!/usr/bin/env python3
"""
Quick Training Visualization Script

Run this anytime during or after training to see:
- How is the loss decreasing?
- Is the bot getting better at poker?
- What's the action distribution?

Usage:
    python Deep_CFR/plot_training.py
    
Or to monitor live:
    watch -n 10 python Deep_CFR/plot_training.py
"""

from training_monitor import create_training_plots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


if __name__ == "__main__":
    print("="*70)
    print("DEEP CFR TRAINING VISUALIZATION")
    print("="*70)

    log_dir = "Deep_CFR/logs"

    # Check if logs exist
    metrics_file = Path(log_dir) / "training_metrics.csv"

    if not metrics_file.exists():
        print(f"\n❌ No training logs found at: {metrics_file}")
        print(f"\nTo start training with monitoring:")
        print(f"  python Deep_CFR/train.py --eval-interval 5")
        sys.exit(1)

    # Count epochs trained
    with open(metrics_file, 'r') as f:
        num_epochs = len(f.readlines()) - 1  # -1 for header

    print(f"\n📊 Found {num_epochs} epochs of training data")
    print(f"📁 Log directory: {log_dir}")
    print(f"\nGenerating plots...")

    try:
        create_training_plots(log_dir)
        print(f"\n✓ Success! View plots at: Deep_CFR/logs/training_progress.png")
    except Exception as e:
        print(f"\n❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
