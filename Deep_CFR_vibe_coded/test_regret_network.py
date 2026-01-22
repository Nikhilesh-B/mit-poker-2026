"""
Test script for RegretNetwork

Demonstrates how the regret network integrates with game states.
"""

import torch
import pkrbot
from regret_network import RegretNetwork, regret_matching, sample_action
from deep_cfr_encoding import encode_state
from action_mapping import get_legal_mask, action_index_to_engine_action, ACTION_NAMES
from skeleton.states import GameState, RoundState, STARTING_STACK, BIG_BLIND, SMALL_BLIND


def create_test_state():
    """Create a simple test game state"""
    deck = pkrbot.Deck()
    deck.shuffle()
    
    hands = [deck.deal(3), deck.deal(3)]
    
    round_state = RoundState(
        button=0,
        street=0,  # Preflop
        pips=[SMALL_BLIND, BIG_BLIND],
        stacks=[STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND],
        hands=hands,
        board=[],
        previous_state=None
    )
    
    game_state = GameState(bankroll=0, game_clock=300.0, round_num=1)
    
    return game_state, round_state


def test_basic_forward_pass():
    """Test 1: Basic network forward pass"""
    print("\n" + "="*70)
    print("TEST 1: Basic Forward Pass")
    print("="*70)
    
    # Create network
    net = RegretNetwork(input_dim=32, output_dim=8)
    print(f"Created network with {net.get_num_parameters():,} parameters")
    
    # Create test state
    game_state, round_state = create_test_state()
    
    # Encode state
    state_encoding = encode_state(game_state, round_state, player_id=0)
    print(f"\nState encoding shape: {state_encoding.shape}")
    
    # Forward pass
    regrets = net(state_encoding)
    print(f"Regrets shape: {regrets.shape}")
    print(f"Regrets: {regrets}")
    
    assert regrets.shape == torch.Size([8]), "Output shape should be [8]"
    print("\n✓ Forward pass successful!")
    
    return net


def test_regret_matching_with_legal_mask():
    """Test 2: Regret matching with legal actions"""
    print("\n" + "="*70)
    print("TEST 2: Regret Matching with Legal Actions")
    print("="*70)
    
    net = RegretNetwork()
    game_state, round_state = create_test_state()
    
    # Encode and predict
    state_encoding = encode_state(game_state, round_state, player_id=0)
    regrets = net(state_encoding)
    
    # Get legal mask
    legal_mask = get_legal_mask(round_state, player_id=0)
    
    print(f"\nLegal actions mask:")
    for i, (is_legal, name) in enumerate(zip(legal_mask, ACTION_NAMES)):
        if is_legal > 0:
            print(f"  ✓ {name}")
    
    # Apply regret matching
    strategy = regret_matching(regrets, legal_mask)
    
    print(f"\nRegrets: {regrets}")
    print(f"\nStrategy after regret matching:")
    for i, (prob, name) in enumerate(zip(strategy, ACTION_NAMES)):
        if prob > 0.01:
            print(f"  {name:20s}: {prob:.4f} ({prob*100:.1f}%)")
    
    # Verify probabilities sum to 1
    prob_sum = strategy.sum().item()
    print(f"\nSum of probabilities: {prob_sum:.6f}")
    assert abs(prob_sum - 1.0) < 1e-5, "Probabilities should sum to 1!"
    
    print("✓ Regret matching works correctly!")
    
    return strategy


def test_action_sampling():
    """Test 3: Sample actions from strategy"""
    print("\n" + "="*70)
    print("TEST 3: Action Sampling")
    print("="*70)
    
    net = RegretNetwork()
    game_state, round_state = create_test_state()
    
    state_encoding = encode_state(game_state, round_state, player_id=0)
    regrets = net(state_encoding)
    legal_mask = get_legal_mask(round_state, player_id=0)
    strategy = regret_matching(regrets, legal_mask)
    
    # Sample multiple actions to see distribution
    samples = []
    num_samples = 1000
    
    print(f"Sampling {num_samples} actions...")
    for _ in range(num_samples):
        action_idx = sample_action(strategy)
        samples.append(action_idx)
    
    # Count frequencies
    print(f"\nEmpirical distribution:")
    for i in range(8):
        count = samples.count(i)
        if count > 0:
            empirical_prob = count / num_samples
            theoretical_prob = strategy[i].item()
            print(f"  {ACTION_NAMES[i]:20s}: {empirical_prob:.4f} "
                  f"(expected {theoretical_prob:.4f})")
    
    print("✓ Action sampling works!")


def test_action_conversion():
    """Test 4: Convert action indices to engine actions"""
    print("\n" + "="*70)
    print("TEST 4: Action Index to Engine Action")
    print("="*70)
    
    game_state, round_state = create_test_state()
    
    # Test each action type
    print("\nTesting action conversions:")
    
    # Fold
    action = action_index_to_engine_action(0, round_state)
    print(f"  Index 0 → {action}")
    
    # Check/Call
    action = action_index_to_engine_action(1, round_state)
    print(f"  Index 1 → {action}")
    
    # Raise
    action = action_index_to_engine_action(2, round_state)
    print(f"  Index 2 → {action}")
    
    print("✓ Action conversion works!")


def test_full_decision_pipeline():
    """Test 5: Complete decision-making pipeline"""
    print("\n" + "="*70)
    print("TEST 5: Full Decision Pipeline")
    print("="*70)
    
    # This simulates one decision point in a game
    net = RegretNetwork()
    game_state, round_state = create_test_state()
    player_id = 0
    
    print(f"\nGame situation:")
    print(f"  Player {player_id}'s hand: {[str(c) for c in round_state.hands[player_id]]}")
    print(f"  Stacks: {round_state.stacks}")
    print(f"  Pips: {round_state.pips}")
    
    # Step 1: Encode state
    state_encoding = encode_state(game_state, round_state, player_id)
    print(f"\n1. Encoded state: {state_encoding.shape}")
    
    # Step 2: Predict regrets
    regrets = net(state_encoding)
    print(f"2. Predicted regrets: {regrets[:3]}... (showing first 3)")
    
    # Step 3: Get legal actions
    legal_mask = get_legal_mask(round_state, player_id)
    print(f"3. Legal mask: {legal_mask}")
    
    # Step 4: Compute strategy
    strategy = regret_matching(regrets, legal_mask)
    print(f"4. Strategy probabilities:")
    for i, (prob, name) in enumerate(zip(strategy, ACTION_NAMES)):
        if prob > 0.01:
            print(f"     {name}: {prob:.4f}")
    
    # Step 5: Sample action
    action_idx = sample_action(strategy)
    print(f"5. Sampled action index: {action_idx} ({ACTION_NAMES[action_idx]})")
    
    # Step 6: Convert to engine action
    engine_action = action_index_to_engine_action(action_idx, round_state)
    print(f"6. Engine action: {engine_action}")
    
    print("\n✓ Full pipeline works end-to-end!")


def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("REGRET NETWORK COMPREHENSIVE TESTS")
    print("="*70)
    
    try:
        test_basic_forward_pass()
        test_regret_matching_with_legal_mask()
        test_action_sampling()
        test_action_conversion()
        test_full_decision_pipeline()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nRegret Network is ready to use!")
        print("\nNext steps:")
        print("  1. Implement CFR traversal (game tree exploration)")
        print("  2. Build training loop (collect data → train network)")
        print("  3. Test on actual poker games")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()

