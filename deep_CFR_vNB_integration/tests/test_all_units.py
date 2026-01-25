"""
Comprehensive Unit Tests for Infoset Parser, MCCFR, and Action Mapping

Run these tests to verify correctness before proceeding to network integration.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import random
import numpy as np
from mccfr import MCCFR
from infoset_parser import (
    parse_infoset_string,
    parse_infoset_to_network_input,
    batch_parse_infosets,
)
from action_mapping import (
    get_all_network_action_keys,
    map_network_output_to_actions,
    apply_legal_action_mask,
    regrets_dict_to_tensor,
    NETWORK_ACTION_TYPES
)
from custom_engine import (
    FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction, TerminalState
)

# Test tracking
tests_passed = 0
tests_failed = 0

def run_test(test_name, test_func):
    global tests_passed, tests_failed
    try:
        test_func()
        print(f"✓ {test_name} PASSED")
        tests_passed += 1
        return True
    except AssertionError as e:
        print(f"✗ {test_name} FAILED: {e}")
        tests_failed += 1
        return False
    except Exception as e:
        print(f"✗ {test_name} ERROR: {e}")
        tests_failed += 1
        return False

print("=" * 70)
print("COMPREHENSIVE UNIT TESTS")
print("=" * 70)

# ===================================================================
# INFOSET PARSER TESTS
# ===================================================================
print("\n[1] INFOSET PARSER TESTS")
print("-" * 70)

def test_parser_simple_preflop():
    infoset = "S0|H:14s0,13s1,10s2|B:|A:"
    cc, ah = parse_infoset_to_network_input(infoset)
    assert cc.canonical_hand == ['14s0', '13s1', '10s2']
    assert cc.canonical_board == []
    assert ah == ""

def test_parser_card_encoding():
    infoset = "S0|H:14s0,13s1,10s2|B:|A:"
    cc, _ = parse_infoset_to_network_input(infoset)
    tensor = cc.get_canonical_hand_tensor()
    assert tensor[0].item() == 140  # 14*10 + 0
    assert tensor[1].item() == 131  # 13*10 + 1
    assert tensor[2].item() == 102  # 10*10 + 2

def test_parser_all_action_types():
    test_cases = [
        ("S0|H:14s0|B:|A:F", "F"),
        ("S0|H:14s0|B:|A:C", "C"),
        ("S0|H:14s0|B:|A:X", "X"),
        ("S0|H:14s0|B:|A:D", "D"),
        ("S0|H:14s0|B:|A:r", "r"),
        ("S0|H:14s0|B:|A:R", "R"),
        ("S0|H:14s0|B:|A:B", "B"),
    ]
    for infoset, expected_ah in test_cases:
        cc, ah = parse_infoset_to_network_input(infoset)
        assert ah == expected_ah

def test_parser_sorting():
    infoset = "S0|H:14s0,10s1,5s2|B:|A:"
    cc, _ = parse_infoset_to_network_input(infoset)
    ranks = [int(card.split('s')[0]) for card in cc.canonical_hand]
    assert ranks == [14, 10, 5]  # Sorted descending

def test_parser_batch_consistency():
    infosets = ["S0|H:14s0|B:|A:", "S0|H:13s0|B:|A:", "S0|H:10s0|B:|A:"]
    cc_list, ah_list = batch_parse_infosets(infosets)
    assert len(cc_list) == 3
    for i, infoset in enumerate(infosets):
        cc_ind, ah_ind = parse_infoset_to_network_input(infoset)
        assert cc_list[i].canonical_hand == cc_ind.canonical_hand

run_test("Parser: Simple preflop", test_parser_simple_preflop)
run_test("Parser: Card encoding", test_parser_card_encoding)
run_test("Parser: All action types", test_parser_all_action_types)
run_test("Parser: Sorting", test_parser_sorting)
run_test("Parser: Batch consistency", test_parser_batch_consistency)

# ===================================================================
# MCCFR CLASS TESTS
# ===================================================================
print("\n[2] MCCFR CLASS TESTS")
print("-" * 70)

def test_mccfr_initial_state():
    random.seed(42)
    mccfr = MCCFR()
    state = mccfr.create_initial_state()
    assert state.street == 0
    assert len(state.hands[0]) == 3
    assert len(state.hands[1]) == 3
    assert len(state.board) == 0
    assert not isinstance(state, TerminalState)

def test_mccfr_infoset_format():
    mccfr = MCCFR()
    state = mccfr.create_initial_state()
    infoset = mccfr.get_infoset(state, 0)
    parts = infoset.split('|')
    assert len(parts) == 4
    assert parts[0].startswith('S')
    assert parts[1].startswith('H:')
    assert parts[2].startswith('B:')
    assert parts[3].startswith('A:')

def test_mccfr_action_keys():
    mccfr = MCCFR()
    state = mccfr.create_initial_state()
    assert mccfr.action_to_key(FoldAction(), state, 0) == "FOLD"
    assert mccfr.action_to_key(CallAction(), state, 0) == "CALL"
    assert mccfr.action_to_key(CheckAction(), state, 0) == "CHECK"

def test_mccfr_regret_matching():
    mccfr = MCCFR()
    state = mccfr.create_initial_state()
    legal_actions = mccfr.get_legal_actions_list(state)
    action_keys = [mccfr.action_to_key(a, state, 0) for a in legal_actions[:3]]
    regrets = {action_keys[0]: 3.0, action_keys[1]: 2.0, action_keys[2]: 1.0}
    strategy = mccfr.regret_matching(regrets, legal_actions[:3], state, 0)
    assert abs(strategy[action_keys[0]] - 0.5) < 0.001
    assert abs(sum(strategy.values()) - 1.0) < 0.001

def test_mccfr_regret_table_growth():
    random.seed(456)
    mccfr = MCCFR()
    assert len(mccfr.regret_table) == 0
    for i in range(5):
        state = mccfr.create_initial_state()
        mccfr.external_sampling(state, i % 2)
    assert len(mccfr.regret_table) > 0

run_test("MCCFR: Initial state", test_mccfr_initial_state)
run_test("MCCFR: Infoset format", test_mccfr_infoset_format)
run_test("MCCFR: Action keys", test_mccfr_action_keys)
run_test("MCCFR: Regret matching", test_mccfr_regret_matching)
run_test("MCCFR: Regret table growth", test_mccfr_regret_table_growth)

# ===================================================================
# ACTION MAPPING TESTS
# ===================================================================
print("\n[3] ACTION MAPPING TESTS")
print("-" * 70)

def test_mapping_network_action_types():
    assert len(NETWORK_ACTION_TYPES) == 9
    assert NETWORK_ACTION_TYPES[3] == 'CHECK'
    assert NETWORK_ACTION_TYPES[4] == 'CALL'
    assert NETWORK_ACTION_TYPES[5] == 'FOLD'

def test_mapping_get_all_keys():
    mccfr = MCCFR()
    state = mccfr.create_initial_state()
    keys = get_all_network_action_keys(state, 0, mccfr)
    assert len(keys) == 9
    assert keys[3] == 'CHECK'
    assert keys[4] == 'CALL'
    assert keys[5] == 'FOLD'

def test_mapping_network_to_dict():
    mccfr = MCCFR()
    state = mccfr.create_initial_state()
    network_output = torch.ones(9) * 10.0
    regret_dict = map_network_output_to_actions(network_output, state, 0, mccfr)
    assert isinstance(regret_dict, dict)
    assert len(regret_dict) > 0

def test_mapping_dict_to_tensor():
    mccfr = MCCFR()
    state = mccfr.create_initial_state()
    regrets_dict = {'FOLD': -10.0, 'CALL': 5.0}
    tensor = regrets_dict_to_tensor(regrets_dict, state, 0, mccfr)
    assert tensor.shape == torch.Size([9])
    assert tensor.dtype == torch.float32
    # Illegal discards should be -1000
    assert tensor[0].item() == -1000.0
    assert tensor[1].item() == -1000.0
    assert tensor[2].item() == -1000.0

def test_mapping_legal_mask():
    mccfr = MCCFR()
    state = mccfr.create_initial_state()
    legal_actions = mccfr.get_legal_actions_list(state)
    regret_dict = {'CALL': 10.0, 'FOLD': 5.0, 'FAKE_ACTION': 100.0}
    masked = apply_legal_action_mask(regret_dict, legal_actions, state, 0, mccfr)
    assert masked.get('FAKE_ACTION', -1000.0) == -1000.0

run_test("Mapping: Network action types", test_mapping_network_action_types)
run_test("Mapping: Get all keys", test_mapping_get_all_keys)
run_test("Mapping: Network to dict", test_mapping_network_to_dict)
run_test("Mapping: Dict to tensor", test_mapping_dict_to_tensor)
run_test("Mapping: Legal mask", test_mapping_legal_mask)

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\nTests passed: {tests_passed}")
print(f"Tests failed: {tests_failed}")
print(f"Total tests: {tests_passed + tests_failed}")
print(f"Pass rate: {100 * tests_passed / (tests_passed + tests_failed):.1f}%")

if tests_failed == 0:
    print("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
    print("\nComponents verified:")
    print("  ✓ Infoset Parser")
    print("  ✓ MCCFR Class")
    print("  ✓ Action Mapping")
    print("\nReady to proceed to Step 4: Test DeepCFR Network")
else:
    print(f"\n✗ {tests_failed} TEST(S) FAILED - Review and fix before proceeding")
    
print("=" * 70)
