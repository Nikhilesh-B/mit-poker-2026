"""
Train Deep CFR Model

This script trains a Deep CFR model and saves it for use in player.py.

Following the paper:
- Two value networks (θ1, θ2) for each player
- Strategy network Π for final play
- Instantaneous regrets with linear weighting
- Networks reinitialized each iteration

Usage:
    python train_model.py --iterations 500 --output deep_cfr_model.pt
"""

import sys
import os
import argparse
import torch

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.deep_cfr import DeepCFR
from training_monitor import TrainingMonitor


def train_model(
    iterations: int = 500,
    network_dim: int = 256,
    learning_rate: float = 0.001,
    batch_size: int = 2000,
    use_network_after: int = 100,
    train_every: int = 10,
    train_epochs: int = 5,
    output_path: str = "deep_cfr_model.pt",
    checkpoint_every: int = 50,
    training_device: str = "auto",
    verbose: bool = True
):
    """
    Train a Deep CFR model and save it.
    
    Args:
        iterations: Number of Deep CFR iterations
        network_dim: Network hidden dimension
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        use_network_after: Start using network after N iterations
        train_every: Train network every N iterations
        train_epochs: Epochs per training session
        output_path: Path to save the model
        checkpoint_every: Save checkpoint every N iterations
        training_device: Device for batch training ("auto", "mps", "cuda", "cpu")
        verbose: Print progress
    """
    # Initialize training monitor
    monitor = TrainingMonitor(
        log_dir="output/logs",
        checkpoint_dir="output/checkpoints"
    )
    
    # Create Deep CFR with paper-aligned settings
    deep_cfr = DeepCFR(
        network_dim=network_dim,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_network_after=use_network_after,
        train_every=train_every,
        train_epochs=train_epochs,
        training_device=training_device
    )
    
    # Get the actual device being used
    actual_device = deep_cfr.trainers[0].training_device
    
    # Start training with monitor
    monitor.start_training(iterations)
    
    print(f"Configuration:")
    print(f"  Network dim: {network_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Training device: {actual_device}")
    print(f"  Train every: {train_every} iterations")
    print(f"  Checkpoint every: {checkpoint_every} iterations")
    print()
    
    # Training loop with monitoring
    for i in range(iterations):
        monitor.start_iteration()
        
        # Run single iteration
        result = deep_cfr.run_iteration()
        
        # Log metrics
        if result.get('trained', False):
            monitor.log_iteration(i + 1, result, verbose=verbose)
            
            # Check if this is the best model
            avg_loss = (result.get('loss_p0', 0) + result.get('loss_p1', 0)) / 2
            is_best = avg_loss > 0 and avg_loss < monitor.best_loss
            
            # Save checkpoint
            if (i + 1) % checkpoint_every == 0:
                monitor.save_checkpoint(deep_cfr, i + 1, is_best=is_best)
    
    # Training complete
    monitor.log_training_complete()
    
    # Save final model
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(f"\nSaving final model to {output_path}...")
    
    # Save the STRATEGY network (this is what player.py should use)
    model_data = {
        'strategy_network_state_dict': deep_cfr.strategy_network.state_dict(),
        'network_p0_state_dict': deep_cfr.networks[0].state_dict(),
        'network_p1_state_dict': deep_cfr.networks[1].state_dict(),
        'network_dim': network_dim,
        'iterations': iterations,
        'final_loss_p0': monitor.history['loss_p0'][-1] if monitor.history['loss_p0'] else None,
        'final_loss_p1': monitor.history['loss_p1'][-1] if monitor.history['loss_p1'] else None,
        'final_loss_strategy': monitor.history['loss_strategy'][-1] if monitor.history['loss_strategy'] else None,
        'training_samples': monitor.get_summary()['total_samples'],
        'history': monitor.history
    }
    
    torch.save(model_data, output_path)
    print(f"✓ Model saved successfully!")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Save training history
    monitor.save_history()
    
    # Generate plot if matplotlib available
    try:
        from training_monitor import plot_training_curves
        plot_path = output_path.replace('.pt', '_training.png')
        plot_training_curves(monitor.history, save_path=plot_path)
    except Exception as e:
        print(f"Could not generate training plot: {e}")
    
    return deep_cfr, model_data


def main():
    parser = argparse.ArgumentParser(description='Train Deep CFR Model (Paper-aligned)')
    parser.add_argument(
        '--iterations', type=int, default=500,
        help='Number of Deep CFR iterations (default: 500)'
    )
    parser.add_argument(
        '--network-dim', type=int, default=256,
        help='Network hidden dimension (default: 256)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=2000,
        help='Batch size for training (default: 2000, paper HULH uses 20000)'
    )
    parser.add_argument(
        '--use-network-after', type=int, default=100,
        help='Start using network after N iterations (default: 100)'
    )
    parser.add_argument(
        '--train-every', type=int, default=10,
        help='Train network every N iterations (default: 10)'
    )
    parser.add_argument(
        '--train-epochs', type=int, default=5,
        help='Epochs per training session (default: 5)'
    )
    parser.add_argument(
        '--output', type=str, default='output/models/deep_cfr_model.pt',
        help='Output model path (default: output/models/deep_cfr_model.pt)'
    )
    parser.add_argument(
        '--checkpoint-every', type=int, default=50,
        help='Save checkpoint every N iterations (default: 50)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'mps', 'cuda', 'cpu'],
        help='Training device: auto (detect MPS/CUDA), mps (Mac GPU), cuda, or cpu (default: auto)'
    )
    
    args = parser.parse_args()
    
    train_model(
        iterations=args.iterations,
        network_dim=args.network_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_network_after=args.use_network_after,
        train_every=args.train_every,
        train_epochs=args.train_epochs,
        output_path=args.output,
        checkpoint_every=args.checkpoint_every,
        training_device=args.device,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
