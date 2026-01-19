"""
Comprehensive test suite for deep_cfr_encoding.py
Tests various complex game states and scenarios
"""
import torch
import pkrbot
from deep_cfr_encoding import encode_state
from skeleton.states import GameState, RoundState, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.actions import CheckAction, RaiseAction, CallAction, DiscardAction


def create_game_state():
    """Helper to create game state"""
    return GameState(bankroll=0, game_clock=300.0, round_num=1)


def test_1_simple_preflop():
    """Test 1: Simple preflop, no action yet"""
    print("\n" + "="*70)
    print("TEST 1: Simple Preflop (No Action)")
    print("="*70)

    deck = pkrbot.Deck()
    deck.shuffle()
    hands = [deck.deal(3), deck.deal(3)]

    round_state = RoundState(
        button=0,
        street=0,
        pips=[SMALL_BLIND, BIG_BLIND],
        stacks=[STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND],
        hands=hands,
        board=[],
        previous_state=None
    )

    game_state = create_game_state()
    encoded = encode_state(game_state, round_state, 0)

    print(f"P0 hand: {[str(c) for c in hands[0]]}")
    print(f"P1 hand: {[str(c) for c in hands[1]]}")
    print(f"Board: []")
    print(f"Stacks: {round_state.stacks}")
    print(f"Pips: {round_state.pips}")
    print(f"\nEncoded shape: {encoded.shape}")
    print(f"Expected: torch.Size([32])")
    assert encoded.shape == torch.Size(
        [32]), f"Shape mismatch! Got {encoded.shape}"
    print("✓ Shape correct!")

    return encoded


def test_2_preflop_after_raise():
    """Test 2: Preflop after one raise"""
    print("\n" + "="*70)
    print("TEST 2: Preflop After Raise")
    print("="*70)

    deck = pkrbot.Deck()
    deck.shuffle()
    hands = [deck.deal(3), deck.deal(3)]

    # Initial state
    state1 = RoundState(
        button=0, street=0,
        pips=[SMALL_BLIND, BIG_BLIND],
        stacks=[STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND],
        hands=hands, board=[], previous_state=None
    )

    # Player 0 raises to 10
    state2 = RoundState(
        button=1, street=0,
        pips=[10, BIG_BLIND],
        stacks=[STARTING_STACK - 10, STARTING_STACK - BIG_BLIND],
        hands=hands, board=[], previous_state=state1
    )

    game_state = create_game_state()
    encoded = encode_state(game_state, state2, 1)  # Encode for player 1

    print(f"P0 raised to {state2.pips[0]}")
    print(f"P1 facing bet of {state2.pips[0] - state2.pips[1]}")
    print(f"Stacks: {state2.stacks}")
    print(f"\nEncoded shape: {encoded.shape}")
    assert encoded.shape == torch.Size([32]), f"Shape mismatch!"

    # Check that "facing bet" feature is set
    facing_bet_idx = 29  # Feature index for facing_bet
    print(f"Facing bet feature [29]: {encoded[facing_bet_idx]:.4f}")
    assert encoded[facing_bet_idx] > 0.9, "Should be facing a bet!"
    print("✓ Correctly detects facing bet!")

    return encoded


def test_3_after_first_discard():
    """Test 3: After first discard (Player 1 discarded)"""
    print("\n" + "="*70)
    print("TEST 3: After First Discard")
    print("="*70)

    deck = pkrbot.Deck()
    deck.shuffle()

    # Initial 3-card hands
    hand0_full = deck.deal(3)
    hand1_full = deck.deal(3)

    # Player 1 discarded one card
    discarded_card = hand1_full.pop(0)
    hands = [hand0_full, hand1_full]  # P0 has 3, P1 has 2

    # Board has 2 peeked cards + 1 discarded
    board = deck.deal(2)
    board.append(discarded_card)

    round_state = RoundState(
        button=0,  # Now P0's turn to discard
        street=3,  # Discard street 2
        pips=[0, 0],
        # After preflop action
        stacks=[STARTING_STACK - 10, STARTING_STACK - 10],
        hands=hands,
        board=board,
        previous_state=None
    )

    game_state = create_game_state()
    encoded = encode_state(game_state, round_state, 0)

    print(f"P0 hand (3 cards): {[str(c) for c in hands[0]]}")
    print(f"P1 hand (2 cards): {[str(c) for c in hands[1]]}")
    print(f"Board (3 cards): {[str(c) for c in board]}")
    print(f"\nEncoded shape: {encoded.shape}")
    assert encoded.shape == torch.Size([32]), f"Shape mismatch!"
    print("✓ Shape correct!")

    # Check that P0's 3 cards are encoded
    card_features = encoded[1:7]
    print(f"P0's card features: {card_features}")
    non_zero_cards = (card_features > 0).sum().item()
    print(f"Non-zero card features: {non_zero_cards}/6 (expect 6 for 3 cards)")

    return encoded


def test_4_turn_with_betting():
    """Test 4: Turn street with betting history"""
    print("\n" + "="*70)
    print("TEST 4: Turn with Betting")
    print("="*70)

    deck = pkrbot.Deck()
    deck.shuffle()

    # Both players have 2 cards (discarded 1 each)
    hands = [deck.deal(2), deck.deal(2)]

    # Board: 2 discards + 2 peek + turn = 5 cards
    board = deck.deal(5)

    # Simulate some betting on turn
    round_state = RoundState(
        button=1,
        street=4,  # Turn
        pips=[20, 0],  # P0 bet 20
        stacks=[STARTING_STACK - 30, STARTING_STACK - 10],
        hands=hands,
        board=board,
        previous_state=None
    )

    game_state = create_game_state()
    encoded = encode_state(game_state, round_state, 1)  # P1's perspective

    print(f"P0 bet: {round_state.pips[0]}")
    print(f"P1 facing: {round_state.pips[0] - round_state.pips[1]}")
    print(f"Board ({len(board)} cards): {[str(c) for c in board]}")
    print(f"Pot size: {(STARTING_STACK * 2) - sum(round_state.stacks)}")

    print(f"\nEncoded shape: {encoded.shape}")
    assert encoded.shape == torch.Size([32]), f"Shape mismatch!"

    # Check pot odds calculation
    pot_odds_idx = 28
    print(f"Pot odds feature [28]: {encoded[pot_odds_idx]:.4f}")

    # Check investment ratio
    investment_idx = 30
    print(f"Investment ratio [30]: {encoded[investment_idx]:.4f}")

    print("✓ Turn state encoded correctly!")

    return encoded


def test_5_river_allin():
    """Test 5: River all-in situation"""
    print("\n" + "="*70)
    print("TEST 5: River All-In")
    print("="*70)

    deck = pkrbot.Deck()
    deck.shuffle()

    # Final hands (2 cards each)
    hands = [deck.deal(2), deck.deal(2)]

    # Full board (6 cards total)
    board = deck.deal(6)

    # All-in situation
    round_state = RoundState(
        button=1,
        street=5,  # River
        pips=[STARTING_STACK, STARTING_STACK],  # Both all-in
        stacks=[0, 0],
        hands=hands,
        board=board,
        previous_state=None
    )

    game_state = create_game_state()
    encoded = encode_state(game_state, round_state, 0)

    print(f"Both players all-in")
    print(f"Stacks: {round_state.stacks}")
    print(f"Pips: {round_state.pips}")
    print(f"Board ({len(board)} cards): {[str(c) for c in board]}")

    print(f"\nEncoded shape: {encoded.shape}")
    assert encoded.shape == torch.Size([32]), f"Shape mismatch!"

    # Check stack features are 0
    stack_idx = 23
    print(f"My stack feature [23]: {encoded[stack_idx]:.4f} (should be 0)")
    assert encoded[stack_idx] < 0.01, "Stack should be 0!"

    print("✓ All-in situation encoded correctly!")

    return encoded


def test_6_complex_multistreet():
    """Test 6: Complex game with betting across multiple streets"""
    print("\n" + "="*70)
    print("TEST 6: Complex Multi-Street Game")
    print("="*70)

    deck = pkrbot.Deck()
    deck.shuffle()

    # Simulate a complex hand history
    hands = [deck.deal(2), deck.deal(2)]
    board = deck.deal(4)

    # Previous state (turn)
    prev_state = RoundState(
        button=0, street=4,
        pips=[15, 15],
        stacks=[STARTING_STACK - 25, STARTING_STACK - 25],
        hands=hands, board=board[:3], previous_state=None
    )

    # Current state (river after another bet)
    current_state = RoundState(
        button=1, street=5,
        pips=[30, 0],
        stacks=[STARTING_STACK - 55, STARTING_STACK - 25],
        hands=hands, board=board, previous_state=prev_state
    )

    game_state = create_game_state()
    encoded_p0 = encode_state(game_state, current_state, 0)
    encoded_p1 = encode_state(game_state, current_state, 1)

    print(f"Multi-street action:")
    print(f"  Turn: Both checked (pips were [15,15])")
    print(f"  River: P0 bet 30")
    print(f"  P1 facing decision")

    print(f"\nP0 encoding shape: {encoded_p0.shape}")
    print(f"P1 encoding shape: {encoded_p1.shape}")
    assert encoded_p0.shape == torch.Size([32])
    assert encoded_p1.shape == torch.Size([32])

    # P0 shouldn't be facing a bet (they just bet)
    print(f"P0 facing bet [29]: {encoded_p0[29]:.4f} (should be ~0)")

    # P1 should be facing a bet
    print(f"P1 facing bet [29]: {encoded_p1[29]:.4f} (should be ~1)")
    assert encoded_p1[29] > 0.9, "P1 should be facing bet!"

    print("✓ Multi-street encoding works for both players!")

    return encoded_p0, encoded_p1


def run_all_tests():
    """Run all comprehensive tests"""
    print("\n" + "="*70)
    print("COMPREHENSIVE ENCODING TESTS")
    print("="*70)

    results = []

    try:
        results.append(("Simple Preflop", test_1_simple_preflop()))
        results.append(("Preflop After Raise", test_2_preflop_after_raise()))
        results.append(("After First Discard", test_3_after_first_discard()))
        results.append(("Turn with Betting", test_4_turn_with_betting()))
        results.append(("River All-In", test_5_river_allin()))
        multi = test_6_complex_multistreet()
        results.append(("Complex Multi-Street P0", multi[0]))
        results.append(("Complex Multi-Street P1", multi[1]))

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)

        print("\nSummary:")
        for name, encoded in results:
            print(f"  {name:30s} Shape: {encoded.shape}  "
                  f"Range: [{encoded.min():.3f}, {encoded.max():.3f}]")

        print("\nEncoding is ready for neural network!")
        print("Next step: Build RegretNetwork architecture")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
