#!/usr/bin/env python3
"""
Quick script to check Deep CFR training progress
"""
import sys
from pathlib import Path
import time

def check_progress():
    checkpoint_dir = Path("../checkpoints")
    
    print("="*70)
    print("DEEP CFR TRAINING PROGRESS")
    print("="*70)
    
    # Check if training is running
    import subprocess
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    is_running = "train.py" in result.stdout
    
    if is_running:
        print("\n✓ Training is ACTIVE")
        # Extract CPU and time info
        for line in result.stdout.split('\n'):
            if 'train.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 10:
                    print(f"  CPU: {parts[2]}% | Memory: {parts[3]}% | Runtime: {parts[9]}")
                    break
    else:
        print("\n✗ Training NOT running")
        print("  Start with: uv run python train.py --epochs 100")
        return
    
    # Check checkpoints
    print("\n" + "-"*70)
    print("CHECKPOINTS SAVED")
    print("-"*70)
    
    if checkpoint_dir.exists():
        epoch_files = sorted(checkpoint_dir.glob("deep_cfr_epoch*.pt"))
        
        if epoch_files:
            print(f"\nFound {len(epoch_files)} checkpoint(s):")
            for f in epoch_files[-5:]:  # Show last 5
                # Extract epoch number
                epoch_num = f.stem.replace("deep_cfr_epoch", "")
                mtime = f.stat().st_mtime
                size_kb = f.stat().st_size / 1024
                time_str = time.strftime("%H:%M:%S", time.localtime(mtime))
                print(f"  • Epoch {epoch_num:>3s} - saved at {time_str} ({size_kb:.1f} KB)")
            
            # Latest epoch
            latest = epoch_files[-1]
            epoch_num = int(latest.stem.replace("deep_cfr_epoch", ""))
            progress = (epoch_num / 100) * 100
            remaining = 100 - epoch_num
            
            print(f"\n📊 Progress: {epoch_num}/100 epochs ({progress:.0f}%)")
            print(f"⏱️  Remaining: ~{remaining} epochs")
            
            # Time estimate
            if epoch_num >= 10:
                # Estimate based on actual progress
                first_checkpoint = epoch_files[0]
                last_checkpoint = epoch_files[-1]
                time_diff = last_checkpoint.stat().st_mtime - first_checkpoint.stat().st_mtime
                epochs_done = epoch_num - int(first_checkpoint.stem.replace("deep_cfr_epoch", ""))
                if epochs_done > 0:
                    time_per_epoch = time_diff / epochs_done
                    remaining_seconds = time_per_epoch * remaining
                    hours = int(remaining_seconds / 3600)
                    minutes = int((remaining_seconds % 3600) / 60)
                    print(f"⏳ Estimated time: {hours}h {minutes}m")
            else:
                print(f"⏳ Estimated time: ~6-7 hours total")
        else:
            print("\nNo checkpoints yet (training started recently)")
            print("First checkpoint will appear after epoch 10 (~40 minutes)")
    
    print("\n" + "-"*70)
    print("NEXT STEPS")
    print("-"*70)
    print("• Monitor: watch -n 30 python3 Deep_CFR/check_progress.py")
    print("• Stop training: pkill -f train.py")
    print("• After completion: uv run python test_trained_model.py")
    print()

if __name__ == "__main__":
    check_progress()

