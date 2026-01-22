"""
Test script for MCCFR player to diagnose bugs and verify correct operation.

This script:
1. Loads the trained strategy
2. Simulates game states
3. Checks if infoset encoding is working
4. Verifies action selection is reasonable
5. Tests for common bugs (encoding mismatches, illegal actions, etc.)
"""

import sys
import os
from pathlib import Path
from collections import defaultdict
import random

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.runner import parse_args, run_bot
from pkrbot import Deck, Card
import pickle

# Import all the functions from player.py
from player import (
    Player, get_infoset, action_to_key, get_legal_actions_list,
    get_card_rank, get_card_suit, canonicalize_cards, get_average_strategy
)

STARTING_STACK = 400


def test_1_load_strategy():
    """Test 1: Verify strategy loads correctly"""
    print("="*80)
    print("TEST 1: Loading Strategy")
    print("="*80)
    
    try:
        with open('mccfr_strategy.pkl', 'rb') as f:
            data = pickle.load(f)
        
        strategy_table = data['strategy_table']
        regret_table = data['regret_table']
        
        print(f"✓ Strategy loaded successfully")
        print(f"  Infosets in strategy table: {len(strategy_table):,}")
        print(f"  Entries in regret table: {len(regret_table):,}")
        
        # Check for different street coverage
        street_counts = defaultdict(int)
        for infoset in list(strategy_table.keys())[:10000]:  # Sample
            if infoset.startswith('S'):
                street = infoset[1]
                street_counts[street] += 1
        
        print(f"\n  Street coverage (sample of 10,000):")
        for street in sorted(street_counts.keys()):
            print(f"    Street {street}: {street_counts[street]:,} infosets")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load strategy: {e}")
        return False


def test_2_canonicalization():
    """Test 2: Verify card canonicalization is working"""
    print("\n" + "="*80)
    print("TEST 2: Card Canonicalization")
    print("="*80)
    
    # Test that equivalent hands produce same canonical form
    test_cases = [
        # (hand, board, description)
        ([Card("As"), Card("Ks"), Card("Qs")], [], "AKQ suited"),
        ([Card("Ah"), Card("Kh"), Card("Qh")], [], "AKQ suited (different suit)"),
        ([Card("2c"), Card("3d"), Card("4s")], [], "234 rainbow"),
    ]
    
    print("Testing suit isomorphism:")
    canonical_hands = []
    for hand, board, desc in test_cases:
        can_hand, can_board = canonicalize_cards(hand, board)
        canonical_hands.append(can_hand)
        print(f"  {desc:30s} → {can_hand}")
    
    # Check if first two (both AKQ suited) are identical
    if canonical_hands[0] == canonical_hands[1]:
        print(f"\n✓ Suit isomorphism working: AKQ♠ == AKQ♥ → {canonical_hands[0]}")
    else:
        print(f"\n✗ BUG: AKQ♠ ≠ AKQ♥")
        print(f"  {canonical_hands[0]} vs {canonical_hands[1]}")
        return False
    
    return True


def test_3_infoset_encoding():
    """Test 3: Check infoset encoding consistency"""
    print("\n" + "="*80)
    print("TEST 3: Infoset Encoding")
    print("="*80)
    
    # Create a simple game state
    deck = Deck()
    deck.shuffle()
    hands = [deck.deal(3), deck.deal(3)]
    
    # Import RoundState properly
    from skeleton.states import RoundState
    
    # Create preflop state - need to check the actual RoundState signature
    # Based on engine.py: RoundState(button, street, pips, stacks, hands, deck, board, previous_state)
    try:
        state = RoundState(1, 0, [1, 2], [399, 398], hands, deck, [], None)
        
        # Get infoset for player 1 (big blind)
        infoset = get_infoset(state, player=1)
        
        print(f"Sample preflop infoset:")
        print(f"  Hand: {[str(c) for c in hands[1]]}")
        print(f"  Infoset: {infoset}")
        
        # Verify infoset format
        if infoset.startswith('S0|H:') and '|B:|A:' in infoset:
            print(f"\n✓ Infoset format correct")
            return True
        else:
            print(f"\n✗ Infoset format incorrect!")
            return False
            
    except Exception as e:
        print(f"✗ Error creating game state: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_action_encoding():
    """Test 4: Verify action encoding is consistent"""
    print("\n" + "="*80)
    print("TEST 4: Action Encoding")
    print("="*80)
    
    deck = Deck()
    deck.shuffle()
    hands = [deck.deal(3), deck.deal(3)]
    
    from skeleton.states import RoundState
    
    try:
        state = RoundState(1, 0, [1, 2], [399, 398], hands, deck, [], None)
        
        # Get legal actions
        legal_actions = get_legal_actions_list(state)
        
        print(f"Legal actions at preflop (facing BB):")
        for action in legal_actions:
            action_key = action_to_key(action, state, active=1)
            print(f"  {type(action).__name__:15s} → {action_key}")
        
        # Check that we have expected actions
        action_types = set(type(a).__name__ for a in legal_actions)
        expected = {'FoldAction', 'CallAction', 'RaiseAction'}
        
        if expected.issubset(action_types):
            print(f"\n✓ Expected preflop actions present")
            return True
        else:
            print(f"\n✗ Missing expected actions!")
            print(f"  Expected: {expected}")
            print(f"  Got: {action_types}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing actions: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_player_decisions():
    """Test 5: Test player makes reasonable decisions"""
    print("\n" + "="*80)
    print("TEST 5: Player Decision Making")
    print("="*80)
    
    # Create player
    player = Player()
    
    if not player.use_mccfr:
        print("✗ Player failed to load strategy!")
        return False
    
    # Simulate 100 random preflop situations
    decisions = defaultdict(int)
    found_in_table = 0
    total_tests = 100
    
    from skeleton.states import RoundState
    
    for i in range(total_tests):
        deck = Deck()
        deck.shuffle()
        hands = [deck.deal(3), deck.deal(3)]
        
        try:
            state = RoundState(1, 0, [1, 2], [399, 398], hands, deck, [], None)
            
            # Check if infoset is in table
            infoset = get_infoset(state, player=1)
            if infoset in player.strategy_table:
                found_in_table += 1
            
            # Get player's action
            action = player.get_action(None, state, active=1)
            
            action_type = type(action).__name__.replace('Action', '').upper()
            decisions[action_type] += 1
            
        except Exception as e:
            print(f"✗ Error in iteration {i+1}: {e}")
            decisions['ERROR'] += 1
    
    print(f"\nTested {total_tests} random preflop situations:")
    print(f"  Found in strategy table: {found_in_table}/{total_tests} ({found_in_table/total_tests*100:.1f}%)")
    print(f"\nAction distribution:")
    
    for action_type, count in sorted(decisions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action_type:10s}: {count:3d} ({count/total_tests*100:.1f}%)")
    
    # Analyze results
    print()
    if decisions.get('ERROR', 0) > 0:
        print(f"✗ Player threw errors during play!")
        return False
    
    if decisions.get('FOLD', 0) > 70:
        print(f"✗ Player folding way too much (>70%)!")
        return False
    
    if found_in_table < 5:
        print(f"⚠️  Very low strategy coverage (<5%) - bot will play mostly random")
        print(f"   This explains poor performance. Need more training.")
    
    print(f"✓ Player making decisions without errors")
    return True


def test_6_strategy_quality():
    """Test 6: Inspect learned strategy quality"""
    print("\n" + "="*80)
    print("TEST 6: Strategy Quality Inspection")
    print("="*80)
    
    with open('mccfr_strategy.pkl', 'rb') as f:
        data = pickle.load(f)
    
    strategy_table = data['strategy_table']
    
    # Get visit counts
    visit_counts = {}
    for infoset, actions in strategy_table.items():
        visit_counts[infoset] = sum(actions.values())
    
    # Sort by visits
    sorted_infosets = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top 5 most visited infosets:")
    for i, (infoset, visits) in enumerate(sorted_infosets[:5], 1):
        print(f"\n{i}. Visits: {visits:.0f}")
        print(f"   {infoset[:70]}...")
        
        # Show strategy
        strategy = strategy_table[infoset]
        total = sum(strategy.values())
        print(f"   Strategy:")
        for action, count in sorted(strategy.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"     {action:20s}: {count/total*100:5.1f}%")
    
    # Statistics
    visit_list = list(visit_counts.values())
    avg_visits = sum(visit_list) / len(visit_list)
    max_visits = max(visit_list)
    
    print(f"\nStrategy statistics:")
    print(f"  Total infosets: {len(strategy_table):,}")
    print(f"  Avg visits per infoset: {avg_visits:.1f}")
    print(f"  Max visits (most common): {max_visits:.0f}")
    
    if avg_visits < 5:
        print(f"\n⚠️  Very sparse strategy (avg <5 visits per state)")
        print(f"   Need significantly more training for good play")
    elif avg_visits < 20:
        print(f"\n⚠️  Low strategy density (avg <20 visits)")
        print(f"   More training recommended")
    else:
        print(f"\n✓ Reasonable strategy density")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "MCCFR PLAYER DIAGNOSTIC TESTS" + " "*29 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        test_1_load_strategy,
        test_2_canonicalization,
        test_3_infoset_encoding,
        test_4_action_encoding,
        test_5_player_decisions,
        test_6_strategy_quality,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  Test {i} ({test.__name__:30s}): {status}")
    
    passed = sum(results)
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n✓ All tests passed! Bot logic is working correctly.")
        print("  If performance is still poor, the issue is insufficient training.")
    else:
        print("\n✗ Some tests failed - bugs detected in bot implementation!")


if __name__ == '__main__':
    main()
