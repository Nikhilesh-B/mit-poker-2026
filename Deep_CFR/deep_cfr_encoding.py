import torch
from skeleton.states import GameState, RoundState, TerminalState
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot


def encode_state(game_state, round_state, player_id):
    '''
    Convert game state to a feature vector for neural network input

    This function transforms a poker game state into a fixed-size numerical
    representation suitable for deep learning. All features are normalized
    to [0,1] range for stable neural network training.

    Features (32 total):
    ┌─────────────────────────────────────────────────────────────┐
    │ Feature Group           | Count | Indices | Description     │
    ├─────────────────────────────────────────────────────────────┤
    │ 1. Hand strength        |   1   |  [0]    | Overall equity  │
    │ 2. My cards             |   6   |  [1-6]  | Rank+suit pairs │
    │ 3. Board cards          |  14   |  [7-20] | Rank+suit pairs │
    │ 4. Game state           |   2   | [21-22] | Street, button  │
    │ 5. Stacks/pot           |   5   | [23-27] | Chip counts     │
    │ 6. Pot odds             |   1   |  [28]   | Call price      │
    │ 7. Betting history      |   3   | [29-31] | Action context  │
    └─────────────────────────────────────────────────────────────┘

    Args:
        game_state: GameState namedtuple (bankroll, clock, round_num)
        round_state: RoundState namedtuple (button, street, pips, stacks, hands, board, previous)
        player_id: int (0 or 1) - which player's perspective to encode

    Returns:
        torch.Tensor of shape [32] with all values in [0,1] range

    Note:
        - All card features are padded with 0.0 if cards were discarded
        - All chip amounts normalized by STARTING_STACK for scale invariance
        - Player-relative encoding: "my" always refers to player_id
    '''
    features = []

    active = player_id
    my_cards = round_state.hands[active]
    board_cards = round_state.board
    street = round_state.street

    my_pip = round_state.pips[active]
    opp_pip = round_state.pips[1-active]
    my_stack = round_state.stacks[active]
    opp_stack = round_state.stacks[1-active]

    continue_cost = opp_pip - my_pip
    my_contribution = STARTING_STACK - my_stack
    opp_contribution = STARTING_STACK - opp_stack
    pot_size = my_contribution + opp_contribution

    # === 1. HAND STRENGTH (1 feature) ===
    # Encodes overall hand quality as a single normalized value
    # TODO: Replace with actual equity calculation from your equity table
    # Current implementation: Simple heuristic based on average card rank
    # - Calculate average rank of all cards in hand
    # - Normalize to [0,1] by dividing by 14 (Ace = 14)
    # - Returns 0.0 if hand is empty (after discards)
    if len(my_cards) > 0:
        avg_rank = sum(card.rank for card in my_cards) / len(my_cards)
        hand_strength = avg_rank / 14.0  # Normalize (2-14 range)
    else:
        hand_strength = 0.0
    features.append(hand_strength)

    # === 2. MY CARDS (6 features: up to 3 cards × 2) ===
    # Encodes the player's hole cards as rank-suit pairs
    # - Each card represented by 2 features: rank and suit
    # - Rank: normalized to [0,1] where 2->0.14, Ace->1.0 (divide by 14)
    # - Suit: normalized to [0,1] where clubs->0, diamonds->0.25, hearts->0.5, spades->0.75
    # - Padding: If cards were discarded, pad with 0.0 to maintain fixed size
    # - Result: Always 6 features (3 cards × 2 features each)
    for i in range(3):
        if i < len(my_cards):
            card = my_cards[i]
            # Normalize rank (2-14) to [0.14, 1.0]
            features.append(card.rank / 14.0)
            # Normalize suit (0-3) to [0, 0.75]
            features.append(card.suit / 4.0)
        else:
            # Padding if card was discarded
            features.append(0.0)
            features.append(0.0)

    # === 3. BOARD CARDS (14 features: up to 7 cards × 2) ===
    # Encodes community cards visible to both players
    # - Board composition: 2 peeked cards + 2 discarded cards + turn + river = max 6 cards
    # - Padded to 7 slots for safety and future expansion
    # - Each card encoded same as hole cards: rank (normalized) + suit (normalized)
    # - Early streets have fewer board cards, padded with 0.0
    # Example progression:
    #   Preflop: all 0s (empty board)
    #   After discards: 4 cards (2 peek + 2 discards)
    #   Turn: 5 cards
    #   River: 6 cards
    for i in range(7):
        if i < len(board_cards):
            card = board_cards[i]
            # Same normalization as hole cards
            features.append(card.rank / 14.0)
            features.append(card.suit / 4.0)
        else:
            # Padding for unused board slots
            features.append(0.0)
            features.append(0.0)

    # === 4. GAME STATE (2 features) ===
    # Encodes high-level game progression and position

    # Feature 1: Street (game phase)
    # - Normalized to [0,1] by dividing by 6 (max street value)
    # - Street values: 0=preflop, 2=discard1, 3=discard2, 4=turn, 5=river, 6=showdown
    # - Example: preflop->0.0, turn->0.67, river->0.83
    features.append(street / 6.0)

    # Feature 2: Button position (who acts last)
    # - Binary value: 0 or 1
    # - Indicates positional advantage
    # - button % 2 gives current position relative to dealer
    features.append(float(round_state.button % 2))

    # === 5. STACKS AND POT (5 features) ===
    # Encodes chip counts and pot size, all normalized by starting stack

    # Feature 1: My remaining stack
    # - Normalized to [0,1] where 0=all-in, 1=full starting stack
    # - Indicates how much we can still bet
    features.append(my_stack / STARTING_STACK)

    # Feature 2: Opponent's remaining stack
    # - Same normalization as my stack
    # - Important for pot odds and fold equity calculations
    features.append(opp_stack / STARTING_STACK)

    # Feature 3: My pips (chips committed this street)
    # - Normalized by starting stack
    # - Resets to 0 at start of each new betting round
    features.append(my_pip / STARTING_STACK)

    # Feature 4: Opponent's pips
    # - Same normalization
    # - Difference between pips tells us bet amount facing
    features.append(opp_pip / STARTING_STACK)

    # Feature 5: Total pot size
    # - Sum of all chips committed by both players
    # - Normalized by total chips in play (2 × STARTING_STACK)
    # - Range: [0, 1] where 1 means all chips in pot
    features.append(pot_size / (STARTING_STACK * 2))

    # === 6. POT ODDS (1 feature) ===
    # Encodes the price being offered to call
    # - Formula: amount_to_call / (pot + amount_to_call)
    # - Represents what % of final pot we must risk to continue
    # - Range: [0, 1]
    # - Examples:
    #   Pot=$100, facing $50 bet: 50/(100+50) = 0.33 (getting 2:1 odds)
    #   Pot=$100, facing $100 bet: 100/(100+100) = 0.5 (getting 1:1 odds)
    # - Returns 0.0 if not facing a bet (no call needed)
    if continue_cost > 0 and (pot_size + continue_cost) > 0:
        pot_odds = continue_cost / (pot_size + continue_cost)
    else:
        pot_odds = 0.0
    features.append(pot_odds)

    # === 7. BETTING HISTORY (3 features) ===
    # Encodes simple betting context for current decision

    # Feature 1: Facing a bet (binary indicator)
    # - 1.0 if opponent has bet/raised and we need to respond
    # - 0.0 if we can check (no bet to call)
    # - Helps network distinguish between proactive vs reactive situations
    facing_bet = 1.0 if continue_cost > 0 else 0.0
    features.append(facing_bet)

    # Feature 2: Investment ratio (pot commitment)
    # - How much of the pot did I contribute?
    # - Formula: my_contribution / total_pot_size
    # - Range: [0, 1] where 0.5 means I put in half the pot
    # - Indicates pot commitment (harder to fold when high)
    # - Returns 0.0 for first action (no pot yet)
    if pot_size > 0:
        my_investment_ratio = my_contribution / pot_size
    else:
        my_investment_ratio = 0.0
    features.append(my_investment_ratio)

    # Feature 3: Pot odds (repeated from feature 6)
    # - Duplicated here for betting history context
    # - TODO: Could replace with different stat like:
    #   - Aggression factor (num raises / num actions)
    #   - Street-specific betting indicator
    #   - Opponent's betting frequency
    features.append(pot_odds)

    # === RETURN FINAL ENCODING ===
    # Convert Python list to PyTorch tensor
    # - dtype=float32 for neural network compatibility
    # - Should have exactly 32 features
    # - All values should be in [0,1] range (some may be exactly 0 or 1)
    assert len(features) == 32, f"Expected 32 features, got {len(features)}"
    return torch.tensor(features, dtype=torch.float32)


# Test function
if __name__ == "__main__":
    print("Testing encode_state function...")

    # Create a mock round state for testing
    # You'll need to actually run this with your real engine
    print(f"\nExpected feature dimension: 32")
    print(f"Feature breakdown:")
    print(f"  1. Hand strength: 1")
    print(f"  2. My cards (3×2): 6")
    print(f"  3. Board cards (7×2): 14")
    print(f"  4. Game state: 2")
    print(f"  5. Stacks/pot: 5")
    print(f"  6. Pot odds: 1")
    print(f"  7. Betting history: 3")
    print(f"  Total: 32")

    print("\nTo test with real game state:")
    print("1. Import your engine and run a game")
    print("2. Call encode_state(game_state, round_state, player_id)")
    print("3. Check that output.shape == torch.Size([33])")
    print("4. Verify all values are in reasonable ranges (mostly 0-1)")

    # Example of how to verify all features are normalized
    print("\nFeature value ranges (should mostly be 0-1):")
    print("  [0]: Hand strength (0-1)")
    print("  [1-6]: My cards (0-1 for rank/suit)")
    print("  [7-20]: Board cards (0-1 for rank/suit)")
    print("  [21-22]: Game state (0-1)")
    print("  [23-27]: Stacks/pot (0-1)")
    print("  [28]: Pot odds (0-1)")
    print("  [29-31]: Betting history (0-1)")
