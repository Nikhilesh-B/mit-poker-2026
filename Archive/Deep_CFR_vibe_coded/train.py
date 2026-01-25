"""
Main Training Script for Deep CFR

This script implements the full Deep CFR training loop:
1. Initialize regret network
2. Run CFR traversals to collect data
3. Train network on collected data
4. Repeat for many epochs
5. Save trained model

Training alternates between:
- Data Collection Phase: Run CFR traversals, collect regret samples
- Training Phase: Update network weights on collected data
"""

import torch
import argparse
import time
from pathlib import Path
import pkrbot

from regret_network import RegretNetwork
from cfr_trainer import CFRTrainer, create_initial_round_state
from skeleton.states import STARTING_STACK, SMALL_BLIND, BIG_BLIND, RoundState
from training_monitor import TrainingMonitor, create_training_plots


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Deep CFR for poker")

    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--traversals-per-epoch", type=int, default=100,
                        help="Number of CFR traversals per epoch")
    parser.add_argument("--train-steps-per-epoch", type=int, default=100,
                        help="Number of network training steps per epoch")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for network")
    parser.add_argument("--buffer-size", type=int, default=100_000,
                        help="Size of training buffer")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 128],
                        help="Hidden layer sizes")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="Evaluate vs baseline every N epochs (0 to disable)")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="Number of games to play for evaluation")

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_deep_cfr(args):
    """
    Main training loop for Deep CFR.

    The training process:
    1. Create regret network
    2. For each epoch:
        a. Data collection: Run many CFR traversals
        b. Network training: Update weights on collected data
        c. Evaluation: Track progress
        d. Checkpointing: Save model periodically
    """
    print("="*70)
    print("DEEP CFR TRAINING")
    print("="*70)

    # Setup
    set_seed(args.seed)
    device = torch.device(args.device)

    # Create network
    print("\nInitializing network...")
    regret_net = RegretNetwork(
        input_dim=32,
        output_dim=8,
        hidden_sizes=tuple(args.hidden_sizes)
    )
    print(f"Network parameters: {regret_net.get_num_parameters():,}")

    # Create trainer
    trainer = CFRTrainer(
        regret_net=regret_net,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=args.device
    )

    # Setup checkpointing
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup training monitor
    monitor = TrainingMonitor(log_dir="Deep_CFR/logs")
    print("✓ Training monitor initialized")
    print(f"  Logs: {monitor.log_dir}")
    print(
        f"  Evaluation: {'Every ' + str(args.eval_interval) + ' epochs' if args.eval_interval > 0 else 'Disabled'}")

    # Training statistics
    start_time = time.time()
    total_traversals = 0

    print("\nStarting training...")
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Traversals per epoch: {args.traversals_per_epoch}")
    print(f"  Train steps per epoch: {args.train_steps_per_epoch}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {args.device}")

    # Main training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()

        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*70}")

        # === PHASE 1: DATA COLLECTION ===
        print(
            f"\nPhase 1: Collecting data ({args.traversals_per_epoch} traversals)...")

        epoch_values_p0 = []
        epoch_values_p1 = []

        for trav_idx in range(args.traversals_per_epoch):
            # Create new random game
            initial_state = create_initial_round_state()

            # Run CFR traversal
            value_p0, value_p1 = trainer.traverse_episode(initial_state)

            epoch_values_p0.append(value_p0)
            epoch_values_p1.append(value_p1)
            total_traversals += 1

            # Progress update
            if (trav_idx + 1) % 20 == 0:
                avg_val = sum(epoch_values_p0[-20:]) / 20
                print(f"  Traversal {trav_idx + 1}/{args.traversals_per_epoch} "
                      f"| Avg P0 value: {avg_val:+.2f}")

        # === PHASE 2: NETWORK TRAINING ===
        print(
            f"\nPhase 2: Training network ({args.train_steps_per_epoch} steps)...")

        train_stats = trainer.train_network(
            num_train_steps=args.train_steps_per_epoch)

        # === STATISTICS ===
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        avg_value_p0 = sum(epoch_values_p0) / len(epoch_values_p0)
        avg_value_p1 = sum(epoch_values_p1) / len(epoch_values_p1)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg P0 value: {avg_value_p0:+.2f}")
        print(f"  Avg P1 value: {avg_value_p1:+.2f}")
        print(f"  Training loss: {train_stats['loss']:.6f}")
        print(f"  Buffer size: {train_stats['samples']}")
        print(f"  Total traversals: {total_traversals}")
        print(f"  Epoch time: {epoch_time:.1f}s")
        print(f"  Total time: {total_time/60:.1f}m")

        # === EVALUATION & MONITORING ===
        action_dist = monitor.get_action_distribution()
        print(f"  Action dist: Fold {action_dist['fold']:.1f}%, "
              f"Call {action_dist['call']:.1f}%, "
              f"Raise {action_dist['raise']:.1f}%, "
              f"Discard {action_dist['discard']:.1f}%")

        # Log metrics to CSV
        monitor.log_epoch_metrics(
            epoch=epoch + 1,
            loss=train_stats['loss'],
            avg_p0_value=avg_value_p0,
            avg_p1_value=avg_value_p1,
            buffer_size=train_stats['samples'],
            action_dist=action_dist
        )
        monitor.reset_action_counts()

        # Evaluate vs baseline every N epochs
        if args.eval_interval > 0 and (epoch + 1) % args.eval_interval == 0:
            eval_results = monitor.evaluate_vs_baseline(
                regret_net=trainer.regret_net,
                num_games=args.eval_games,
                device=args.device
            )
            monitor.log_evaluation(
                epoch=epoch + 1,
                num_games=args.eval_games,
                results=eval_results
            )

        # === CHECKPOINTING ===
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = save_dir / f"deep_cfr_epoch{epoch + 1}.pt"
            save_checkpoint(trainer, epoch, checkpoint_path, args)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")

    # === FINAL SAVE ===
    final_path = save_dir / "deep_cfr_final.pt"
    save_checkpoint(trainer, args.epochs, final_path, args)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Total traversals: {total_traversals}")
    print(f"Final model saved: {final_path}")

    # Generate training plots
    print(f"\nGenerating training visualizations...")
    try:
        create_training_plots(log_dir="Deep_CFR/logs")
    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")

    print(f"\nNext steps:")
    print(f"  1. Check training plots: Deep_CFR/logs/training_progress.png")
    print(f"  2. Review metrics: Deep_CFR/logs/training_metrics.csv")
    print(f"  3. Test the model using test_trained_model.py")
    print(f"  4. Use it in player.py for actual games")
    print(f"  5. Continue training with more epochs if needed")


def save_checkpoint(trainer: CFRTrainer, epoch: int, path: Path, args):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'regret_net_state_dict': trainer.regret_net.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'training_stats': trainer.get_stats(),
        'args': vars(args)
    }, path)


def load_checkpoint(path: str, device: str = "cpu"):
    """Load a saved checkpoint"""
    checkpoint = torch.load(path, map_location=device)

    # Reconstruct network
    args = argparse.Namespace(**checkpoint['args'])
    regret_net = RegretNetwork(
        input_dim=32,
        output_dim=8,
        hidden_sizes=tuple(args.hidden_sizes)
    )
    regret_net.load_state_dict(checkpoint['regret_net_state_dict'])
    regret_net.to(device)
    regret_net.eval()

    return regret_net, checkpoint


if __name__ == "__main__":
    args = parse_args()
    train_deep_cfr(args)
