"""
Test a trained Deep CFR model

This script loads a trained model and demonstrates:
1. Loading a checkpoint
2. Running inference on game states
3. Evaluating the learned strategy
"""

import torch
import pkrbot
import argparse
from pathlib import Path

from train import load_checkpoint
from cfr_trainer import create_initial_round_state
from deep_cfr_encoding import encode_state
from regret_network import regret_matching, sample_action
from action_mapping import get_legal_mask, ACTION_NAMES


def test_model(checkpoint_path: str):
    """
    Test a trained Deep CFR model.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
    """
    print("="*70)
    print("TESTING TRAINED DEEP CFR MODEL")
    print("="*70)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    regret_net, checkpoint = load_checkpoint(checkpoint_path)
    
    print(f"\nCheckpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Network params: {regret_net.get_num_parameters():,}")
    print(f"  Training stats: {checkpoint['training_stats']}")
    
    # Create a test game state
    print("\n" + "-"*70)
    print("Testing inference on random game states...")
    print("-"*70)
    
    for game_num in range(3):
        print(f"\nGame {game_num + 1}:")
        
        # Create random initial state
        initial_state = create_initial_round_state()
        active_player = initial_state.button % 2
        
        print(f"  Street: {initial_state.street}")
        print(f"  Active player: {active_player}")
        print(f"  Player {active_player} hand: {initial_state.hands[active_player]}")
        print(f"  Stacks: P0={initial_state.stacks[0]}, P1={initial_state.stacks[1]}")
        
        # Encode state
        state_encoding = encode_state(None, initial_state, active_player)
        
        # Get legal mask
        legal_mask = get_legal_mask(initial_state, active_player)
        legal_actions = [ACTION_NAMES[i] for i in range(8) if legal_mask[i] > 0]
        print(f"  Legal actions: {legal_actions}")
        
        # Get regret predictions
        with torch.no_grad():
            predicted_regrets = regret_net(state_encoding)
        
        # Convert to strategy
        strategy = regret_matching(predicted_regrets, legal_mask)
        
        # Show strategy probabilities
        print(f"\n  Strategy (action probabilities):")
        for i in range(8):
            if legal_mask[i] > 0:
                prob = strategy[i].item()
                regret = predicted_regrets[i].item()
                print(f"    {ACTION_NAMES[i]:20s}: {prob*100:5.1f}% (regret: {regret:+.2f})")
        
        # Sample action
        action_idx = sample_action(strategy)
        print(f"\n  Sampled action: {ACTION_NAMES[action_idx]}")
    
    print("\n" + "="*70)
    print("✓ Model inference working!")
    print("="*70)
    print("\nThe trained model can:")
    print("  • Encode game states into feature vectors")
    print("  • Predict regrets for each action")
    print("  • Convert regrets to action probabilities (strategy)")
    print("  • Sample actions from the learned strategy")
    print("\n💡 This model can now be integrated into player.py for actual games!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained Deep CFR model")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/deep_cfr_final.pt",
                       help="Path to checkpoint file")
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print(f"Please train a model first using train.py")
        exit(1)
    
    test_model(str(checkpoint_path))

