"""
Train Deep CFR Model

This script trains a Deep CFR model and saves it for use in player.py.

Usage:
    python train_model.py --iterations 500 --output deep_cfr_model.pt
"""

import sys
import os
import argparse
import torch

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from deep_cfr import DeepCFR


def train_model(
    iterations: int = 500,
    network_dim: int = 256,
    learning_rate: float = 0.001,
    use_network_after: int = 100,
    train_every: int = 10,
    train_epochs: int = 5,
    output_path: str = "deep_cfr_model.pt",
    verbose: bool = True
):
    """
    Train a Deep CFR model and save it.
    
    Args:
        iterations: Number of Deep CFR iterations
        network_dim: Network hidden dimension
        learning_rate: Learning rate for training
        use_network_after: Start using network after N iterations
        train_every: Train network every N iterations
        train_epochs: Epochs per training session
        output_path: Path to save the model
        verbose: Print progress
    """
    print("=" * 70)
    print("TRAINING DEEP CFR MODEL")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Iterations: {iterations}")
    print(f"  Network dim: {network_dim}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Use network after: {use_network_after} iterations")
    print(f"  Train every: {train_every} iterations")
    print(f"  Train epochs: {train_epochs}")
    print(f"  Output: {output_path}")
    print()
    
    # Create Deep CFR
    deep_cfr = DeepCFR(
        network_dim=network_dim,
        learning_rate=learning_rate,
        use_network_after=use_network_after,
        train_every=train_every,
        train_epochs=train_epochs
    )
    
    # Train
    print("Starting training...")
    results = deep_cfr.run_multiple_iterations(iterations, verbose=verbose)
    
    # Get final statistics
    stats = deep_cfr.get_stats()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal Statistics:")
    print(f"  Total iterations: {len(stats['iterations'])}")
    print(f"  Network usages: {sum(stats['network_usage'])}")
    print(f"  Training sessions: {len(stats['losses'])}")
    if len(stats['losses']) > 0:
        print(f"  Initial loss: {stats['losses'][0]:,.0f}")
        print(f"  Final loss: {stats['losses'][-1]:,.0f}")
        if len(stats['losses']) > 1:
            improvement = ((stats['losses'][0] - stats['losses'][-1]) / stats['losses'][0]) * 100
            print(f"  Loss improvement: {improvement:.1f}%")
    if len(stats['sample_counts']) > 0:
        print(f"  Training samples: {stats['sample_counts'][-1]}")
    
    # Save model
    print(f"\nSaving model to {output_path}...")
    model_data = {
        'network_state_dict': deep_cfr.network.state_dict(),
        'network_dim': network_dim,
        'iterations': iterations,
        'final_loss': stats['losses'][-1] if len(stats['losses']) > 0 else None,
        'training_samples': stats['sample_counts'][-1] if len(stats['sample_counts']) > 0 else 0,
        'stats': stats
    }
    
    torch.save(model_data, output_path)
    print(f"✓ Model saved successfully!")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return deep_cfr, model_data


def main():
    parser = argparse.ArgumentParser(description='Train Deep CFR Model')
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
        '--output', type=str, default='deep_cfr_model.pt',
        help='Output model path (default: deep_cfr_model.pt)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    train_model(
        iterations=args.iterations,
        network_dim=args.network_dim,
        learning_rate=args.learning_rate,
        use_network_after=args.use_network_after,
        train_every=args.train_every,
        train_epochs=args.train_epochs,
        output_path=args.output,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
