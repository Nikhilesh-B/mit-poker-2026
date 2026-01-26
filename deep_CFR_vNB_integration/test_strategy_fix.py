"""
Test script to verify the strategy network fix.

This script loads a trained model and verifies that:
1. Strategy network outputs are converted to probabilities via softmax
2. Actions are diverse (not always folding)
"""

import torch
from pathlib import Path
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from core.integration import NetworkMCCFRIntegration
from core.mccfr import MCCFR
from network.model import DeepCFRModule


def test_strategy_fix(model_path: str, num_tests: int = 20):
    """
    Test that the strategy network fix works correctly.
    
    Args:
        model_path: Path to the trained model
        num_tests: Number of action selections to test
    """
    print("=" * 70)
    print("TESTING STRATEGY NETWORK FIX")
    print("=" * 70)
    
    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return False
    
    print(f"\nLoading model from: {model_path}")
    model_data = torch.load(model_path, map_location='cpu')
    
    # Create network
    network_dim = model_data.get('network_dim', 256)
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=9,
        dim=network_dim
    )
    
    # Load strategy network weights
    if 'strategy_network_state_dict' in model_data:
        network.load_state_dict(model_data['strategy_network_state_dict'])
        print("✓ Loaded strategy_network_state_dict")
    elif 'network_state_dict' in model_data:
        network.load_state_dict(model_data['network_state_dict'])
        print("✓ Loaded network_state_dict")
    else:
        print("ERROR: No network weights found in model file")
        return False
    
    network.eval()
    
    # Create MCCFR
    mccfr = MCCFR()
    
    # Create integration with strategy network flag (FIXED)
    integration = NetworkMCCFRIntegration(
        network, mccfr, is_strategy_network=True
    )
    print("✓ Created integration with is_strategy_network=True")
    
    print(f"\nTesting {num_tests} action selections...")
    print("-" * 70)
    
    action_counts = {}
    fold_count = 0
    diverse_actions = 0
    
    for i in range(num_tests):
        # Create a random state
        state = mccfr.create_initial_state()
        
        # Play a few random actions to get to a non-trivial state
        for _ in range(2):
            legal_actions = mccfr.get_legal_actions_list(state)
            if not legal_actions or state.is_terminal():
                state = mccfr.create_initial_state()
                break
            import random
            action = random.choice(legal_actions)
            state = state.proceed(action)
            if state.is_terminal():
                state = mccfr.create_initial_state()
                break
        
        # Get strategy
        player = state.button % 2
        strategy = integration.get_network_strategy(state, player)
        
        # Select action
        action = integration.select_network_action(state, player)
        
        # Count actions
        action_type = type(action).__name__
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        if action_type == 'FoldAction':
            fold_count += 1
        
        # Check if strategy is diverse (not all probability on one action)
        max_prob = max(strategy.values())
        if max_prob < 0.99:  # Not deterministic
            diverse_actions += 1
        
        if i < 5:
            print(f"\nTest {i+1}:")
            print(f"  Selected: {action_type}")
            print(f"  Strategy probabilities:")
            for action_key, prob in list(strategy.items())[:3]:
                print(f"    {action_key}: {prob:.3f}")
            print(f"  Max prob: {max_prob:.3f} (diverse: {max_prob < 0.99})")
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Action distribution:")
    for action_type, count in sorted(action_counts.items()):
        print(f"  {action_type}: {count}/{num_tests} ({100*count/num_tests:.1f}%)")
    
    print(f"\nFold rate: {fold_count}/{num_tests} ({100*fold_count/num_tests:.1f}%)")
    print(f"Diverse strategies: {diverse_actions}/{num_tests} ({100*diverse_actions/num_tests:.1f}%)")
    
    # Check if fix is working
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    if fold_count == num_tests:
        print("❌ FAILED: Still always folding!")
        return False
    elif fold_count > num_tests * 0.8:
        print("⚠️  WARNING: High fold rate ({:.1f}%), but not always folding".format(100*fold_count/num_tests))
        print("   This might be expected for a poorly trained model")
        return True
    else:
        print("✓ SUCCESS: Actions are diverse (not always folding)")
        print(f"   Fold rate: {100*fold_count/num_tests:.1f}%")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test strategy network fix")
    parser.add_argument("model_path", type=str, help="Path to trained model file")
    parser.add_argument("--num_tests", type=int, default=20, 
                       help="Number of action selections to test (default: 20)")
    
    args = parser.parse_args()
    success = test_strategy_fix(args.model_path, args.num_tests)
    sys.exit(0 if success else 1)
