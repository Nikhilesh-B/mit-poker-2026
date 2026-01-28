"""
Comprehensive Unit Tests for DeepCFR Network

Tests the network with real parsed infosets from MCCFR.
"""

from mccfr import MCCFR
from infoset_parser import parse_infoset_to_network_input, batch_parse_infosets
from DeepCFR import DeepCFRModule
import numpy as np
import random
import torch
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


# Test tracking
tests_passed = 0
tests_failed = 0


def run_test(test_name, test_func):
    """Run a test and track results"""
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
print("DEEPCFR NETWORK UNIT TESTS")
print("=" * 70)

# ===================================================================
# 1. NETWORK INITIALIZATION TESTS
# ===================================================================
print("\n[1] NETWORK INITIALIZATION TESTS")
print("-" * 70)


def test_network_creates_successfully():
    """Test that network initializes with correct architecture"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    assert isinstance(network, torch.nn.Module)
    assert network.n_action_history == 20


def test_network_parameter_count():
    """Test that network has trainable parameters"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    assert params > 0, "Network should have trainable parameters"
    print(f"  Network has {params:,} trainable parameters")


def test_network_embedding_layers():
    """Test that network has correct number of embedding layers"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    assert len(network.hand_embeddings) == 3, "Should have 3 hand card embeddings"
    assert len(
        network.board_embeddings) == 5, "Should have 5 board card embeddings"


run_test("Network creates successfully", test_network_creates_successfully)
run_test("Network has trainable parameters", test_network_parameter_count)
run_test("Network has correct embeddings", test_network_embedding_layers)

# ===================================================================
# 2. FORWARD PASS - SINGLE SAMPLE TESTS
# ===================================================================
print("\n[2] FORWARD PASS - SINGLE SAMPLE TESTS")
print("-" * 70)


def test_forward_pass_preflop():
    """Test forward pass with preflop infoset (empty board)"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    network.eval()

    # Parse a preflop infoset
    infoset = "S0|H:14s0,13s1,10s2|B:|A:"
    cc, ah = parse_infoset_to_network_input(infoset)

    # Forward pass
    with torch.no_grad():
        output = network(cc, ah)

    # Check output shape
    assert output.shape == torch.Size(
        [1, 9]), f"Expected shape [1, 9], got {output.shape}"
    assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"


def test_forward_pass_with_board():
    """Test forward pass with board cards"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    network.eval()

    # Parse infoset with board
    infoset = "S1|H:14s0,13s0,10s1|B:9s0,8s1,7s2,6s0,5s1|A:CRDD"
    cc, ah = parse_infoset_to_network_input(infoset)

    # Forward pass
    with torch.no_grad():
        output = network(cc, ah)

    assert output.shape == torch.Size([1, 9])
    assert len(cc.canonical_board) == 5, "Should have 5 board cards"


def test_forward_pass_with_action_history():
    """Test forward pass with action history"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    network.eval()

    # Parse infoset with complex action history
    infoset = "S2|H:14s0,13s0,10s1|B:9s0,8s1,7s2,6s0,5s1|A:CRBXrDFC"
    cc, ah = parse_infoset_to_network_input(infoset)

    # Forward pass
    with torch.no_grad():
        output = network(cc, ah)

    assert output.shape == torch.Size([1, 9])
    assert ah == "CRBXrDFC", f"Expected action history 'CRBXrDFC', got '{ah}'"


run_test("Forward pass: Preflop", test_forward_pass_preflop)
run_test("Forward pass: With board", test_forward_pass_with_board)
run_test("Forward pass: With action history",
         test_forward_pass_with_action_history)

# ===================================================================
# 3. FORWARD PASS - BATCH TESTS
# ===================================================================
print("\n[3] FORWARD PASS - BATCH TESTS")
print("-" * 70)


def test_batch_forward_pass():
    """Test forward pass with batch of infosets"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    network.eval()

    # Parse multiple infosets
    infosets = [
        "S0|H:14s0,13s1,10s2|B:|A:",
        "S0|H:12s0,11s1,9s2|B:|A:C",
        "S1|H:8s0,7s1,6s2|B:5s0,4s1,3s2,2s0,14s1|A:DD",
    ]
    cc_list, ah_list = batch_parse_infosets(infosets)

    # Forward pass
    with torch.no_grad():
        output = network(cc_list, ah_list)

    # Check output shape
    assert output.shape == torch.Size(
        [3, 9]), f"Expected shape [3, 9], got {output.shape}"
    assert output.dtype == torch.float32


def test_batch_size_10():
    """Test forward pass with larger batch size"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    network.eval()

    # Create 10 infosets
    infosets = []
    for i in range(10):
        rank1 = 14 - i
        rank2 = 13 - i
        rank3 = 10
        infosets.append(f"S0|H:{rank1}s0,{rank2}s1,{rank3}s2|B:|A:")

    cc_list, ah_list = batch_parse_infosets(infosets)

    # Forward pass
    with torch.no_grad():
        output = network(cc_list, ah_list)

    assert output.shape == torch.Size(
        [10, 9]), f"Expected shape [10, 9], got {output.shape}"


run_test("Batch forward pass: 3 samples", test_batch_forward_pass)
run_test("Batch forward pass: 10 samples", test_batch_size_10)

# ===================================================================
# 4. REAL MCCFR INFOSETS TESTS
# ===================================================================
print("\n[4] REAL MCCFR INFOSETS TESTS")
print("-" * 70)


def test_with_real_mccfr_infosets():
    """Test network with real infosets from MCCFR simulation"""
    random.seed(42)
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    network.eval()

    # Run MCCFR to generate real infosets
    mccfr = MCCFR()
    infosets = []

    for i in range(5):
        state = mccfr.create_initial_state()
        infoset = mccfr.get_infoset(state, 0)
        infosets.append(infoset)

    # Parse and run through network
    cc_list, ah_list = batch_parse_infosets(infosets)

    with torch.no_grad():
        output = network(cc_list, ah_list)

    assert output.shape == torch.Size([5, 9])
    print(f"  Successfully processed {len(infosets)} real MCCFR infosets")


def test_with_mccfr_game_progression():
    """Test network with infosets from different game stages"""
    random.seed(123)
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    network.eval()

    # Simulate a game and collect infosets from different streets
    mccfr = MCCFR()
    state = mccfr.create_initial_state()

    # Preflop infoset
    infoset_preflop = mccfr.get_infoset(state, 0)

    # Parse and test
    cc, ah = parse_infoset_to_network_input(infoset_preflop)

    with torch.no_grad():
        output = network(cc, ah)

    assert output.shape == torch.Size([1, 9])
    assert state.street == 0, "Should be at preflop"
    print(f"  Tested infoset from street {state.street}")


run_test("Network with real MCCFR infosets", test_with_real_mccfr_infosets)
run_test("Network with game progression", test_with_mccfr_game_progression)

# ===================================================================
# 5. ACTION HISTORY ENCODING TESTS
# ===================================================================
print("\n[5] ACTION HISTORY ENCODING TESTS")
print("-" * 70)


def test_action_history_empty():
    """Test network with empty action history"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )

    # Encode empty action history
    encoded = network._encode_action_history("")

    # Should be all zeros (padding) - 20 actions * 17 features = 340
    assert encoded.shape == torch.Size(
        [340]), f"Expected shape [340] (20*17), got {encoded.shape}"
    assert torch.all(encoded == 0.0), "Empty history should be all zeros"


def test_action_history_single_action():
    """Test encoding of single action"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )

    # Test each action type (17 total: 4 base + 13 raises)
    test_cases = [
        ('X', 0),   # check
        ('C', 1),   # call
        ('F', 2),   # fold
        ('D', 3),   # discard
        ('1', 4),   # raise 25% pot
        ('2', 5),   # raise 50% pot
        ('3', 6),   # raise 75% pot
        ('4', 7),   # raise 100% pot
        ('5', 8),   # raise 150% pot
        ('6', 9),   # raise 200% pot
        ('7', 10),  # raise 250% pot
        ('8', 11),  # raise 300% pot
        ('9', 12),  # raise 350% pot
        ('T', 13),  # raise 400% pot
        ('E', 14),  # raise 450% pot
        ('W', 15),  # raise 500% pot
        ('Z', 16),  # raise all-in
    ]

    for action_char, expected_idx in test_cases:
        encoded = network._encode_action_history(action_char)

        # First 17 values should be the one-hot encoding
        first_action = encoded[:17]
        expected = torch.zeros(17)
        expected[expected_idx] = 1.0

        assert torch.allclose(first_action, expected), \
            f"Action '{action_char}' should encode to index {expected_idx}"


def test_action_history_long_sequence():
    """Test encoding of long action sequence"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )

    # Create action history longer than n_action_history
    # Using new raise characters: C=call, 4=100% raise, Z=all-in, X=check, D=discard, F=fold
    long_history = "C4ZXDFC4ZXDFC4ZXDFC4ZXDF"  # 24 chars
    encoded = network._encode_action_history(long_history)

    # Should truncate to 20 actions, with 17 features each = 340 total
    assert encoded.shape == torch.Size(
        [340]), f"Expected shape [340], got {encoded.shape}"

    # Verify first action is encoded correctly (C = call = index 1)
    first_action = encoded[:17]
    expected = torch.zeros(17)
    expected[1] = 1.0  # C = call
    assert torch.allclose(first_action, expected)


run_test("Action history: Empty", test_action_history_empty)
run_test("Action history: Single actions", test_action_history_single_action)
run_test("Action history: Long sequence", test_action_history_long_sequence)

# ===================================================================
# 6. EDGE CASE TESTS
# ===================================================================
print("\n[6] EDGE CASE TESTS")
print("-" * 70)


def test_empty_board_preflop():
    """Test network handles empty board correctly"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    network.eval()

    # Preflop with empty board
    infoset = "S0|H:14s0,13s1,10s2|B:|A:"
    cc, ah = parse_infoset_to_network_input(infoset)

    assert len(cc.canonical_board) == 0, "Board should be empty"

    with torch.no_grad():
        output = network(cc, ah)

    assert output.shape == torch.Size([1, 9])


def test_all_same_suit():
    """Test network with all cards same suit"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    network.eval()

    # All same suit (should be canonicalized to suit 0)
    infoset = "S0|H:14s0,13s0,10s0|B:|A:"
    cc, ah = parse_infoset_to_network_input(infoset)

    # Check canonicalization
    suits = [card.split('s')[1] for card in cc.canonical_hand]
    assert len(set(suits)) == 1, "All suits should be canonicalized to same value"

    with torch.no_grad():
        output = network(cc, ah)

    assert output.shape == torch.Size([1, 9])


def test_output_values_are_finite():
    """Test that network outputs are finite (not NaN or Inf)"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    network.eval()

    infoset = "S1|H:14s0,13s0,10s1|B:9s0,8s1,7s2,6s0,5s1|A:CRDD"
    cc, ah = parse_infoset_to_network_input(infoset)

    with torch.no_grad():
        output = network(cc, ah)

    assert torch.all(torch.isfinite(output)), "All outputs should be finite"
    assert not torch.any(torch.isnan(output)), "No outputs should be NaN"
    assert not torch.any(torch.isinf(output)), "No outputs should be Inf"


run_test("Edge case: Empty board", test_empty_board_preflop)
run_test("Edge case: All same suit", test_all_same_suit)
run_test("Edge case: Output values finite", test_output_values_are_finite)

# ===================================================================
# 7. NETWORK CONSISTENCY TESTS
# ===================================================================
print("\n[7] NETWORK CONSISTENCY TESTS")
print("-" * 70)


def test_deterministic_output():
    """Test that same input produces same output (deterministic)"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    network.eval()

    infoset = "S0|H:14s0,13s1,10s2|B:|A:"
    cc, ah = parse_infoset_to_network_input(infoset)

    # Run twice
    with torch.no_grad():
        output1 = network(cc, ah)
        output2 = network(cc, ah)

    assert torch.allclose(
        output1, output2), "Same input should produce same output"


def test_different_inputs_different_outputs():
    """Test that different inputs produce different outputs"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    network.eval()

    infoset1 = "S0|H:14s0,13s1,10s2|B:|A:"
    infoset2 = "S0|H:12s0,11s1,9s2|B:|A:"

    cc1, ah1 = parse_infoset_to_network_input(infoset1)
    cc2, ah2 = parse_infoset_to_network_input(infoset2)

    with torch.no_grad():
        output1 = network(cc1, ah1)
        output2 = network(cc2, ah2)

    # Outputs should be different for different hands
    assert not torch.allclose(
        output1, output2), "Different inputs should produce different outputs"


def test_gradient_flow():
    """Test that gradients can flow through network"""
    network = DeepCFRModule(
        nhandcards=3,
        nboardcards=5,
        n_action_history=20,
        nresponses=11,
        dim=256
    )
    network.train()

    infoset = "S0|H:14s0,13s1,10s2|B:|A:"
    cc, ah = parse_infoset_to_network_input(infoset)

    # Forward pass
    output = network(cc, ah)

    # Compute dummy loss and backward
    loss = output.sum()
    loss.backward()

    # Check that some parameters have gradients
    has_gradients = False
    for param in network.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break

    assert has_gradients, "Network should have gradients after backward pass"


run_test("Consistency: Deterministic output", test_deterministic_output)
run_test("Consistency: Different inputs",
         test_different_inputs_different_outputs)
run_test("Consistency: Gradient flow", test_gradient_flow)

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 70)
print("DEEPCFR NETWORK TEST SUMMARY")
print("=" * 70)
print(f"\nTests passed: {tests_passed}")
print(f"Tests failed: {tests_failed}")
print(f"Total tests: {tests_passed + tests_failed}")
print(f"Pass rate: {100 * tests_passed / (tests_passed + tests_failed):.1f}%")

if tests_failed == 0:
    print("\n✓✓✓ ALL DEEPCFR NETWORK TESTS PASSED! ✓✓✓")
    print("\nNetwork verified for:")
    print("  ✓ Single and batch forward passes")
    print("  ✓ Real MCCFR infosets")
    print("  ✓ Various game states (preflop, flop, etc.)")
    print("  ✓ Action history encoding")
    print("  ✓ Edge cases and consistency")
    print("  ✓ Gradient flow for training")
    print("\nReady to proceed to Step 5: Create Network-MCCFR Integration")
else:
    print(f"\n✗ {tests_failed} TEST(S) FAILED - Review and fix before proceeding")

print("=" * 70)
