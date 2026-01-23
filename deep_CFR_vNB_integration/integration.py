"""
Minimal Network-MCCFR Integration

This module connects the DeepCFR network with MCCFR for a single state.
It demonstrates the full pipeline: state → network → regret matching → action selection.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
from typing import Dict, List, Tuple
from DeepCFR import DeepCFRModule
from infoset_parser import parse_infoset_to_network_input
from action_mapping import (
    map_network_output_to_actions,
    apply_legal_action_mask,
    regrets_dict_to_tensor
)
from mccfr import MCCFR


class NetworkMCCFRIntegration:
    """
    Integrates DeepCFR network with MCCFR algorithm.
    
    This class provides methods to:
    1. Get network predictions for a game state
    2. Convert predictions to MCCFR-compatible regrets
    3. Use regret matching to select actions
    4. Compare network predictions vs tabular MCCFR
    """
    
    def __init__(self, network: DeepCFRModule, mccfr: MCCFR):
        """
        Initialize integration.
        
        Args:
            network: DeepCFR network for regret prediction
            mccfr: MCCFR instance for game logic
        """
        self.network = network
        self.mccfr = mccfr
        self.network.eval()  # Set to evaluation mode
    
    def get_network_regrets(self, state, player: int) -> Dict[str, float]:
        """
        Get network-predicted regrets for a state.
        
        Args:
            state: Current RoundState
            player: Player index (0 or 1)
        
        Returns:
            Dictionary mapping action_key -> regret value
        """
        # Step 1: Get infoset string
        infoset = self.mccfr.get_infoset(state, player)
        
        # Step 2: Parse infoset for network
        cc, ah = parse_infoset_to_network_input(infoset)
        
        # Step 3: Get network prediction
        with torch.no_grad():
            network_output = self.network(cc, ah)  # Shape: [1, 9]
        
        # Step 4: Convert to MCCFR action keys
        regret_dict = map_network_output_to_actions(
            network_output[0],  # Remove batch dimension
            state,
            player,
            self.mccfr
        )
        
        # Step 5: Apply legal action mask
        legal_actions = self.mccfr.get_legal_actions_list(state)
        regret_dict = apply_legal_action_mask(
            regret_dict,
            legal_actions,
            state,
            player,
            self.mccfr
        )
        
        return regret_dict
    
    def get_network_strategy(self, state, player: int) -> Dict[str, float]:
        """
        Get strategy (action probabilities) using network regrets.
        
        Args:
            state: Current RoundState
            player: Player index (0 or 1)
        
        Returns:
            Dictionary mapping action_key -> probability
        """
        # Get network regrets
        regret_dict = self.get_network_regrets(state, player)
        
        # Get legal actions
        legal_actions = self.mccfr.get_legal_actions_list(state)
        
        # Apply regret matching
        strategy = self.mccfr.regret_matching(
            regret_dict,
            legal_actions,
            state,
            player
        )
        
        return strategy
    
    def select_network_action(self, state, player: int):
        """
        Select an action using network predictions.
        
        Args:
            state: Current RoundState
            player: Player index (0 or 1)
        
        Returns:
            Selected action instance
        """
        # Get strategy from network
        strategy = self.get_network_strategy(state, player)
        
        # Sample action according to strategy
        legal_actions = self.mccfr.get_legal_actions_list(state)
        action_keys = [self.mccfr.action_to_key(a, state, player) 
                      for a in legal_actions]
        
        # Create probability distribution
        probs = [strategy.get(key, 0.0) for key in action_keys]
        
        # Normalize if needed
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            # Uniform if no valid probabilities
            probs = [1.0 / len(probs)] * len(probs)
        
        # Sample action
        import random
        selected_idx = random.choices(range(len(legal_actions)), weights=probs)[0]
        return legal_actions[selected_idx]
    
    def compare_network_vs_tabular(self, state, player: int) -> Dict:
        """
        Compare network predictions with tabular MCCFR regrets.
        
        Args:
            state: Current RoundState
            player: Player index (0 or 1)
        
        Returns:
            Dictionary with comparison metrics
        """
        infoset = self.mccfr.get_infoset(state, player)
        
        # Get network regrets
        network_regrets = self.get_network_regrets(state, player)
        
        # Get tabular regrets
        tabular_regrets = self.mccfr.regret_table[infoset]
        
        # Get legal actions
        legal_actions = self.mccfr.get_legal_actions_list(state)
        
        # Compute strategies
        network_strategy = self.mccfr.regret_matching(
            network_regrets, legal_actions, state, player
        )
        
        if tabular_regrets:
            tabular_strategy = self.mccfr.regret_matching(
                tabular_regrets, legal_actions, state, player
            )
        else:
            # Uniform if no tabular data
            action_keys = [self.mccfr.action_to_key(a, state, player) 
                          for a in legal_actions]
            tabular_strategy = {key: 1.0 / len(action_keys) 
                               for key in action_keys}
        
        return {
            'infoset': infoset,
            'network_regrets': network_regrets,
            'tabular_regrets': dict(tabular_regrets),
            'network_strategy': network_strategy,
            'tabular_strategy': tabular_strategy,
            'legal_actions': [self.mccfr.action_to_key(a, state, player) 
                             for a in legal_actions]
        }


def test_single_state_integration():
    """
    Test network integration on a single state.
    
    This is the simplest integration test - can the network
    make predictions for a single game state?
    """
    print("=" * 70)
    print("TESTING SINGLE STATE INTEGRATION")
    print("=" * 70)
    
    # Create network
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=9,
        dim=256
    )
    
    # Create MCCFR
    mccfr = MCCFR()
    state = mccfr.create_initial_state()
    
    # Create integration
    integration = NetworkMCCFRIntegration(network, mccfr)
    
    # Test 1: Get network regrets
    print("\n[1] Getting network regrets...")
    regrets = integration.get_network_regrets(state, player=0)
    print(f"✓ Network regrets: {len(regrets)} actions")
    for action_key, regret in list(regrets.items())[:3]:
        print(f"    {action_key}: {regret:.3f}")
    
    # Test 2: Get network strategy
    print("\n[2] Getting network strategy...")
    strategy = integration.get_network_strategy(state, player=0)
    print(f"✓ Network strategy: {len(strategy)} actions")
    for action_key, prob in list(strategy.items())[:3]:
        print(f"    {action_key}: {prob:.3f}")
    
    # Verify probabilities sum to 1
    total_prob = sum(strategy.values())
    print(f"    Total probability: {total_prob:.6f}")
    assert abs(total_prob - 1.0) < 0.001, f"Probabilities should sum to 1, got {total_prob}"
    
    # Test 3: Select action
    print("\n[3] Selecting action...")
    action = integration.select_network_action(state, player=0)
    print(f"✓ Selected action: {type(action).__name__}")
    
    # Test 4: Compare with tabular (empty at start)
    print("\n[4] Comparing network vs tabular...")
    comparison = integration.compare_network_vs_tabular(state, player=0)
    print(f"✓ Comparison complete")
    print(f"    Legal actions: {len(comparison['legal_actions'])}")
    print(f"    Network strategy: {len(comparison['network_strategy'])} actions")
    print(f"    Tabular strategy: {len(comparison['tabular_strategy'])} actions")
    
    print("\n" + "=" * 70)
    print("✓✓✓ SINGLE STATE INTEGRATION TEST PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_single_state_integration()
