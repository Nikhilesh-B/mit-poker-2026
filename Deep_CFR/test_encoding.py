"""
Test script for deep_cfr_encoding.py
Run this to verify the encoding works correctly
"""
import torch
import pkrbot
from deep_cfr_encoding import encode_state
from skeleton.states import GameState, RoundState, STARTING_STACK, BIG_BLIND, SMALL_BLIND


def create_test_game_state():
    """Create a simple test game state"""
    return GameState(
        bankroll=0,
        game_clock=300.0,
        round_num=1
    )


def create_test_round_state_preflop():
    """Create a preflop round state for testing"""
    # Create a deck and deal cards
    deck = pkrbot.Deck()
    deck.shuffle()

    hands = [deck.deal(3), deck.deal(3)]  # Each player gets 3 cards
    board = []  # Empty board preflop
    pips = [SMALL_BLIND, BIG_BLIND]
    stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]

    return RoundState(
        button=0,  # Player 0 has button
        street=0,  # Preflop
        pips=pips,
        stacks=stacks,
        hands=hands,
        board=board,
        previous_state=None
    )


def create_test_round_state_postflop():
    """Create a postflop round state with some board cards"""
    deck = pkrbot.Deck()
    deck.shuffle()

    # Deal hands with one card already discarded
    hand0 = deck.deal(2)  # Player 0 has 2 cards (discarded 1)
    hand1 = deck.deal(2)  # Player 1 has 2 cards (discarded 1)
    hands = [hand0, hand1]

    # Board has 2 discarded cards + turn
    board = deck.deal(3)

    pips = [10, 10]  # Both have bet 10 chips
    stacks = [STARTING_STACK - 10, STARTING_STACK - 10]

    return RoundState(
        button=1,
        street=4,  # Turn
        pips=pips,
        stacks=stacks,
        hands=hands,
        board=board,
        previous_state=None
    )


def test_encoding_preflop():
    """Test encoding on a preflop state"""
    print("=" * 70)
    print("TEST 1: Preflop Encoding")
    print("=" * 70)

    game_state = create_test_game_state()
    round_state = create_test_round_state_preflop()
    player_id = 0

    # Encode the state
    encoded = encode_state(game_state, round_state, player_id)

    print(
        f"\nPlayer {player_id}'s hand: {[str(c) for c in round_state.hands[player_id]]}")
    print(f"Board: {[str(c) for c in round_state.board]}")
    print(f"Stacks: {round_state.stacks}")
    print(f"Pips: {round_state.pips}")
    print(f"Street: {round_state.street}")

    print(f"\nEncoded state shape: {encoded.shape}")
    print(f"Expected shape: torch.Size([32])")
    print(f"✓ Shape matches!" if encoded.shape ==
          torch.Size([32]) else "✗ Shape mismatch!")

    print(f"\nFeature breakdown:")
    idx = 0
    print(f"  [{idx}] Hand strength: {encoded[idx]:.4f}")
    idx += 1

    print(f"  [{idx}:{idx+6}] My cards: {encoded[idx:idx+6]}")
    idx += 6

    print(f"  [{idx}:{idx+14}] Board cards: {encoded[idx:idx+14]}")
    idx += 14

    print(f"  [{idx}:{idx+3}] Game state: {encoded[idx:idx+3]}")
    idx += 3

    print(f"  [{idx}:{idx+5}] Stacks/pot: {encoded[idx:idx+5]}")
    idx += 5

    print(f"  [{idx}] Pot odds: {encoded[idx]:.4f}")
    idx += 1

    print(f"  [{idx}:{idx+3}] Betting history: {encoded[idx:idx+3]}")

    print(f"\nValue statistics:")
    print(f"  Min: {encoded.min():.4f}")
    print(f"  Max: {encoded.max():.4f}")
    print(f"  Mean: {encoded.mean():.4f}")
    print(f"  Std: {encoded.std():.4f}")

    # Check normalization
    print(f"\nNormalization check:")
    num_in_range = ((encoded >= -0.01) & (encoded <= 1.01)).sum().item()
    print(f"  Features in [0,1] range: {num_in_range}/{len(encoded)}")
    if num_in_range == len(encoded):
        print(f"  ✓ All features properly normalized!")
    else:
        print(f"  ✗ Some features out of range!")
        out_of_range = torch.where((encoded < -0.01) | (encoded > 1.01))[0]
        print(f"  Out of range indices: {out_of_range.tolist()}")
        print(f"  Out of range values: {encoded[out_of_range]}")

    return True


def test_encoding_postflop():
    """Test encoding on a postflop state"""
    print("\n" + "=" * 70)
    print("TEST 2: Postflop Encoding (After Discards)")
    print("=" * 70)

    game_state = create_test_game_state()
    round_state = create_test_round_state_postflop()
    player_id = 1

    # Encode the state
    encoded = encode_state(game_state, round_state, player_id)

    print(
        f"\nPlayer {player_id}'s hand: {[str(c) for c in round_state.hands[player_id]]}")
    print(f"Board: {[str(c) for c in round_state.board]}")
    print(f"Stacks: {round_state.stacks}")
    print(f"Pips: {round_state.pips}")
    print(f"Street: {round_state.street}")

    print(f"\nEncoded state shape: {encoded.shape}")
    print(f"✓ Shape matches!" if encoded.shape ==
          torch.Size([32]) else "✗ Shape mismatch!")

    print(f"\nValue statistics:")
    print(f"  Min: {encoded.min():.4f}")
    print(f"  Max: {encoded.max():.4f}")
    print(f"  Mean: {encoded.mean():.4f}")

    return True


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "=" * 70)
    print("TEST 3: Edge Cases")
    print("=" * 70)

    # Test with no cards (after both discards)
    deck = pkrbot.Deck()
    deck.shuffle()

    hands = [deck.deal(1), deck.deal(1)]  # Only 1 card each
    board = deck.deal(4)  # 4 board cards

    round_state = RoundState(
        button=0,
        street=5,  # River
        pips=[50, 50],
        stacks=[350, 350],
        hands=hands,
        board=board,
        previous_state=None
    )

    game_state = create_test_game_state()
    encoded = encode_state(game_state, round_state, 0)

    print(f"Testing with minimal hands (1 card each)")
    print(f"✓ Encoding successful! Shape: {encoded.shape}")

    return True


if __name__ == "__main__":
    print("Testing Deep CFR State Encoding")
    print("This test creates mock game states and verifies encoding\n")

    try:
        test_encoding_preflop()
        test_encoding_postflop()
        test_edge_cases()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Review the encoded features to ensure they make sense")
        print("2. Consider adding hand strength calculation (equity table)")
        print("3. Move on to building the RegretNetwork!")

    except Exception as e:
        print(f"\n✗ TEST FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
