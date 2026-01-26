"""
Training Monitor for Deep CFR

Provides:
- Real-time logging during training
- Loss tracking and visualization
- Checkpoint saving/loading
- Progress display with ETA
"""

import os
import time
import json
import torch
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class TrainingMonitor:
    """
    Monitor and log Deep CFR training progress.

    Features:
    - Track losses for value networks (P0, P1) and strategy network
    - Track sample counts and training time
    - Save/load checkpoints
    - Generate training plots
    """

    def __init__(self, log_dir: str = "output/logs", checkpoint_dir: str = "output/checkpoints"):
        """
        Initialize training monitor.

        Args:
            log_dir: Directory for log files
            checkpoint_dir: Directory for model checkpoints
        """
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training history
        self.history = {
            'iterations': [],
            'loss_p0': [],
            'loss_p1': [],
            'loss_strategy': [],
            'samples_p0': [],
            'samples_p1': [],
            'samples_strategy': [],
            'time_per_iter': [],
            'timestamps': [],
            # Loss progression within each training session
            'loss_start_strategy': [],
            'loss_end_strategy': [],
            'loss_reduction_strategy': []
        }

        # Timing
        self.start_time = None
        self.iter_start_time = None
        self.total_iterations = 0

        # Best model tracking
        self.best_loss = float('inf')
        self.best_iteration = 0

    def start_training(self, total_iterations: int):
        """Start training session."""
        self.start_time = time.time()
        self.total_iterations = total_iterations
        print(f"\n{'='*60}")
        print(f"Deep CFR Training Started")
        print(f"{'='*60}")
        print(f"Total iterations: {total_iterations}")
        print(f"Log directory: {self.log_dir}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"{'='*60}\n")

    def start_iteration(self):
        """Mark start of an iteration."""
        self.iter_start_time = time.time()

    def log_iteration(self, iteration: int, metrics: Dict, verbose: bool = True):
        """
        Log metrics for an iteration.

        Args:
            iteration: Current iteration number
            metrics: Dictionary with training metrics
            verbose: Whether to print to console
        """
        # Calculate iteration time
        iter_time = time.time() - self.iter_start_time if self.iter_start_time else 0

        # Store history
        self.history['iterations'].append(iteration)
        self.history['loss_p0'].append(metrics.get('loss_p0', 0.0))
        self.history['loss_p1'].append(metrics.get('loss_p1', 0.0))
        self.history['loss_strategy'].append(metrics.get('loss_strategy', 0.0))
        self.history['samples_p0'].append(metrics.get('total_samples_p0', 0))
        self.history['samples_p1'].append(metrics.get('total_samples_p1', 0))
        self.history['samples_strategy'].append(
            metrics.get('strategy_samples', 0))
        self.history['time_per_iter'].append(iter_time)
        self.history['timestamps'].append(datetime.now().isoformat())
        # Loss progression for strategy network
        self.history['loss_start_strategy'].append(
            metrics.get('loss_start_strategy'))
        self.history['loss_end_strategy'].append(
            metrics.get('loss_end_strategy'))
        self.history['loss_reduction_strategy'].append(
            metrics.get('loss_reduction_strategy'))

        # Track best model
        avg_loss = (metrics.get('loss_p0', 0) + metrics.get('loss_p1', 0)) / 2
        if avg_loss > 0 and avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.best_iteration = iteration

        # Print progress
        if verbose:
            self._print_progress(iteration, metrics, iter_time)

    def _print_progress(self, iteration: int, metrics: Dict, iter_time: float):
        """Print formatted progress line with loss progression."""
        # Calculate ETA
        elapsed = time.time() - self.start_time
        avg_time = elapsed / iteration if iteration > 0 else 0
        remaining = avg_time * (self.total_iterations - iteration)
        eta = timedelta(seconds=int(remaining))

        # Format losses (average)
        loss_p0 = metrics.get('loss_p0', 0)
        loss_p1 = metrics.get('loss_p1', 0)
        loss_strat = metrics.get('loss_strategy', 0)

        # Loss progression (start → end)
        loss_start_strat = metrics.get('loss_start_strategy')
        loss_end_strat = metrics.get('loss_end_strategy')
        loss_reduction_strat = metrics.get('loss_reduction_strategy')

        # Format sample counts (in K)
        samples_p0 = metrics.get('total_samples_p0', 0) / 1000
        samples_p1 = metrics.get('total_samples_p1', 0) / 1000
        samples_strat = metrics.get('strategy_samples', 0) / 1000

        # Progress bar
        progress = iteration / self.total_iterations
        bar_width = 20
        filled = int(bar_width * progress)
        bar = '█' * filled + '░' * (bar_width - filled)

        # Print training results on a clean line
        print(f"[TRAIN] Iter {iteration:4d}/{self.total_iterations} [{bar}] {progress*100:5.1f}% | "
              f"Loss V0: {loss_p0:8.0f} V1: {loss_p1:8.0f} | "
              f"Samples: {samples_p0:.0f}K/{samples_p1:.0f}K/{samples_strat:.0f}K | "
              f"ETA: {eta}", flush=True)

        # Print strategy network loss progression (the key metric to watch)
        if loss_start_strat is not None and loss_end_strat is not None:
            reduction_str = f"↓{loss_reduction_strat:.1f}%" if loss_reduction_strat and loss_reduction_strat > 0 else f"↑{abs(loss_reduction_strat or 0):.1f}%"
            print(
                f"        Π (Strategy): {loss_start_strat:,.0f} → {loss_end_strat:,.0f} ({reduction_str})", flush=True)

    def log_training_complete(self):
        """Log training completion."""
        total_time = time.time() - self.start_time

        print(f"\n\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Total time: {timedelta(seconds=int(total_time))}")
        print(f"Iterations: {len(self.history['iterations'])}")
        print(
            f"Best loss: {self.best_loss:.2f} at iteration {self.best_iteration}")

        if self.history['loss_p0']:
            print(f"\nFinal losses:")
            print(f"  Value P0: {self.history['loss_p0'][-1]:.2f}")
            print(f"  Value P1: {self.history['loss_p1'][-1]:.2f}")
            print(f"  Strategy: {self.history['loss_strategy'][-1]:.2f}")

        if self.history['samples_p0']:
            print(f"\nFinal sample counts:")
            print(f"  P0: {self.history['samples_p0'][-1]:,}")
            print(f"  P1: {self.history['samples_p1'][-1]:,}")
            print(f"  Strategy: {self.history['samples_strategy'][-1]:,}")

        print(f"{'='*60}\n")

    def save_checkpoint(self, deep_cfr, iteration: int, is_best: bool = False, 
                        output_path: str = None):
        """
        Save model checkpoint.

        Args:
            deep_cfr: DeepCFR instance
            iteration: Current iteration
            is_best: Whether this is the best model so far
            output_path: Base output path (e.g., 'output/models/my_model.pt')
                        Checkpoints will be named 'my_model_iter_50.pt', etc.
        """
        checkpoint = {
            'iteration': iteration,
            'network_p0_state': deep_cfr.networks[0].state_dict(),
            'network_p1_state': deep_cfr.networks[1].state_dict(),
            'strategy_network_state': deep_cfr.strategy_network.state_dict(),
            'history': self.history,
            'best_loss': self.best_loss,
            'best_iteration': self.best_iteration,
            'config': {
                'network_dim': deep_cfr.network_dim,
                'batch_size': deep_cfr.batch_size,
                'learning_rate': deep_cfr.learning_rate,
            }
        }

        # Save periodic checkpoint (for resuming training)
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_iter_{iteration}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"\n  [Saved best model at iteration {iteration}]")

        # Save latest
        latest_path = os.path.join(self.checkpoint_dir, "latest_model.pt")
        torch.save(checkpoint, latest_path)
        
        # Save player-compatible model file (can be used directly with engine.py --model)
        if output_path:
            # Create name based on output path: my_model.pt -> my_model_iter_50.pt
            base_path = output_path.rsplit('.pt', 1)[0]
            playable_path = f"{base_path}_iter_{iteration}.pt"
            
            playable_model = {
                'strategy_network_state_dict': deep_cfr.strategy_network.state_dict(),
                'network_p0_state_dict': deep_cfr.networks[0].state_dict(),
                'network_p1_state_dict': deep_cfr.networks[1].state_dict(),
                'network_dim': deep_cfr.network_dim,
                'iterations': iteration,
                'training_samples': self.history['samples_strategy'][-1] if self.history['samples_strategy'] else 0,
            }
            torch.save(playable_model, playable_path)
            print(f"  [Saved playable model: {playable_path}]")

    def load_checkpoint(self, deep_cfr, checkpoint_path: str) -> int:
        """
        Load model checkpoint.

        Args:
            deep_cfr: DeepCFR instance to load into
            checkpoint_path: Path to checkpoint file

        Returns:
            Iteration number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path)

        deep_cfr.networks[0].load_state_dict(checkpoint['network_p0_state'])
        deep_cfr.networks[1].load_state_dict(checkpoint['network_p1_state'])
        deep_cfr.strategy_network.load_state_dict(
            checkpoint['strategy_network_state'])

        self.history = checkpoint.get('history', self.history)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_iteration = checkpoint.get('best_iteration', 0)

        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        return checkpoint['iteration']

    def save_history(self, filename: str = "training_history.json"):
        """Save training history to JSON file."""
        history_path = os.path.join(self.log_dir, filename)
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {history_path}")

    def get_summary(self) -> Dict:
        """Get training summary."""
        return {
            'total_iterations': len(self.history['iterations']),
            'final_loss_p0': self.history['loss_p0'][-1] if self.history['loss_p0'] else 0,
            'final_loss_p1': self.history['loss_p1'][-1] if self.history['loss_p1'] else 0,
            'final_loss_strategy': self.history['loss_strategy'][-1] if self.history['loss_strategy'] else 0,
            'best_loss': self.best_loss,
            'best_iteration': self.best_iteration,
            'total_samples': (
                (self.history['samples_p0'][-1] if self.history['samples_p0'] else 0) +
                (self.history['samples_p1'][-1] if self.history['samples_p1'] else 0) +
                (self.history['samples_strategy'][-1]
                 if self.history['samples_strategy'] else 0)
            )
        }


def plot_training_curves(history: Dict, save_path: str = None):
    """
    Plot training curves.

    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot generation.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    iterations = history['iterations']

    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(iterations, history['loss_p0'], label='Value P0', alpha=0.7)
    ax1.plot(iterations, history['loss_p1'], label='Value P1', alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Value Network Losses')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Strategy loss with progression (start → end)
    ax2 = axes[0, 1]
    ax2.plot(iterations, history['loss_strategy'],
             label='Avg Loss', color='green', alpha=0.7, linewidth=2)
    # Plot start and end losses if available
    if 'loss_start_strategy' in history and history['loss_start_strategy']:
        loss_start = [
            x if x is not None else 0 for x in history['loss_start_strategy']]
        loss_end = [
            x if x is not None else 0 for x in history['loss_end_strategy']]
        if any(x > 0 for x in loss_start):
            ax2.plot(iterations, loss_start, label='Start Loss',
                     color='red', alpha=0.5, linestyle='--')
            ax2.plot(iterations, loss_end, label='End Loss',
                     color='blue', alpha=0.5, linestyle='--')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Strategy Network Loss (Π)\n(Start→End shows SGD progress)')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Sample counts
    ax3 = axes[1, 0]
    ax3.plot(iterations, [s/1000 for s in history['samples_p0']],
             label='P0 Samples', alpha=0.7)
    ax3.plot(iterations, [s/1000 for s in history['samples_p1']],
             label='P1 Samples', alpha=0.7)
    ax3.plot(iterations, [s/1000 for s in history['samples_strategy']],
             label='Strategy Samples', alpha=0.7)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Samples (K)')
    ax3.set_title('Training Samples')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Time per iteration
    ax4 = axes[1, 1]
    ax4.plot(iterations, history['time_per_iter'], alpha=0.7)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Time per Iteration')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training plot to {save_path}")


if __name__ == "__main__":
    # Test the monitor
    monitor = TrainingMonitor()

    print("Training Monitor Test")
    print("=" * 50)

    # Simulate training
    monitor.start_training(total_iterations=10)

    for i in range(1, 11):
        monitor.start_iteration()
        time.sleep(0.1)  # Simulate work

        metrics = {
            'loss_p0': 1000000 / (i + 1),
            'loss_p1': 1000000 / (i + 1),
            'loss_strategy': 500000 / (i + 1),
            'total_samples_p0': i * 1000,
            'total_samples_p1': i * 1000,
            'strategy_samples': i * 2000,
        }

        monitor.log_iteration(i, metrics)

    monitor.log_training_complete()

    print("\n✓ Training monitor working!")
