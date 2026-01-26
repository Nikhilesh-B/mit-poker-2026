"""
Diagnostic script to inspect strategy network outputs.

This script loads a trained model and inspects what the network is actually
outputting to understand why the player is always folding.
"""

import torch
import torch.nn.functional as F
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
from utils.infoset_parser import parse_infoset_to_network_input
from utils.action_mapping import map_network_output_to_actions, apply_legal_action_mask


def inspect_model_outputs(model_path: str, num_states: int = 10):
    """
    Inspect what the strategy network is outputting.
    
    Args:
        model_path: Path to the trained model
        num_states: Number of random states to inspect
    """
    print("=" * 70)
    print("STRATEGY NETWORK DIAGNOSTIC")
    print("=" * 70)
    
    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return
    
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
        return
    
    network.eval()
    
    # Create MCCFR for game states
    mccfr = MCCFR()
    
    # Create integration (treating it as strategy network - correct behavior)
    integration = NetworkMCCFRIntegration(network, mccfr, is_strategy_network=True)
    
    print(f"\nInspecting {num_states} random game states...")
    print("-" * 70)
    
    all_negative_count = 0
    all_positive_count = 0
    mixed_count = 0
    always_fold_count = 0
    
    for i in range(num_states):
        # Create a random state by playing some actions
        state = mccfr.create_initial_state()
        
        # Play a few random actions to get to a non-trivial state
        for _ in range(3):
            legal_actions = mccfr.get_legal_actions_list(state)
            if not legal_actions:
                break
            import random
            action = random.choice(legal_actions)
            state = state.proceed(action)
            if state.is_terminal():
                state = mccfr.create_initial_state()
                break
        
        # Get infoset
        player = state.button % 2
        infoset = mccfr.get_infoset(state, player)
        
        # Get raw network output (logits)
        cc, ah = parse_infoset_to_network_input(infoset)
        with torch.no_grad():
            raw_output = network(cc, ah)[0]  # Remove batch dimension
        
        # Convert to action dict
        regret_dict = map_network_output_to_actions(
            raw_output, state, player, mccfr
        )
        
        legal_actions = mccfr.get_legal_actions_list(state)
        regret_dict = apply_legal_action_mask(
            regret_dict, legal_actions, state, player, mccfr
        )
        
        # Get strategy via regret matching (current buggy behavior)
        strategy = mccfr.regret_matching(
            regret_dict, legal_actions, state, player
        )
        
        # Apply softmax to raw outputs (correct behavior per paper)
        raw_tensor = torch.tensor([regret_dict.get(mccfr.action_to_key(a, state, player), 0.0) 
                                  for a in legal_actions])
        softmax_probs = F.softmax(raw_tensor, dim=0)
        softmax_dict = {mccfr.action_to_key(a, state, player): float(softmax_probs[i])
                        for i, a in enumerate(legal_actions)}
        
        # Analyze outputs
        raw_values = list(regret_dict.values())
        all_negative = all(v <= 0 for v in raw_values)
        all_positive = all(v >= 0 for v in raw_values)
        mixed = not all_negative and not all_positive
        
        if all_negative:
            all_negative_count += 1
        elif all_positive:
            all_positive_count += 1
        else:
            mixed_count += 1
        
        # Check if regret matching leads to always fold
        fold_key = mccfr.action_to_key(
            next((a for a in legal_actions if isinstance(a, type(mccfr.create_initial_state().legal_actions()[0]))), None),
            state, player
        )
        # Actually, let's check the strategy directly
        max_prob_action = max(strategy.items(), key=lambda x: x[1])
        always_fold = max_prob_action[1] > 0.99 and 'fold' in max_prob_action[0].lower()
        
        if always_fold:
            always_fold_count += 1
        
        # Print details for first few states
        if i < 3:
            print(f"\nState {i+1}:")
            print(f"  Infoset: {infoset[:60]}...")
            print(f"  Raw outputs (logits): {[f'{v:.3f}' for v in raw_values[:5]]}...")
            print(f"  All negative: {all_negative}, All positive: {all_positive}, Mixed: {mixed}")
            print(f"  Regret matching strategy:")
            for action_key, prob in list(strategy.items())[:3]:
                print(f"    {action_key}: {prob:.3f}")
            print(f"  Softmax strategy (correct):")
            for action_key, prob in list(softmax_dict.items())[:3]:
                print(f"    {action_key}: {prob:.3f}")
            print(f"  Max prob action (regret matching): {max_prob_action[0]} ({max_prob_action[1]:.3f})")
            max_softmax = max(softmax_dict.items(), key=lambda x: x[1])
            print(f"  Max prob action (softmax): {max_softmax[0]} ({max_softmax[1]:.3f})")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"States with all negative outputs: {all_negative_count}/{num_states}")
    print(f"States with all positive outputs: {all_positive_count}/{num_states}")
    print(f"States with mixed outputs: {mixed_count}/{num_states}")
    print(f"States where regret matching → always fold: {always_fold_count}/{num_states}")
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if all_negative_count > num_states * 0.7:
        print("⚠️  BUG CONFIRMED: Most outputs are negative!")
        print("   When all regrets are negative, regret matching:")
        print("   - Clamps all to 0")
        print("   - Sum = 0 → picks highest (least negative) action with prob 1")
        print("   - This leads to deterministic (often folding) behavior")
        print("\n   SOLUTION: Apply softmax to strategy network outputs")
        print("   (Per paper Section 5.1: outputs are logits → softmax → probabilities)")
    else:
        print("✓ Outputs are not all negative, but may still have issues")
        print("  Check if regret matching is appropriate for strategy network")
    
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose strategy network outputs")
    parser.add_argument("model_path", type=str, help="Path to trained model file")
    parser.add_argument("--num_states", type=int, default=10, 
                       help="Number of states to inspect (default: 10)")
    
    args = parser.parse_args()
    inspect_model_outputs(args.model_path, args.num_states)
