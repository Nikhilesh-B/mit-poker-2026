"""
MCCFR Player Bot

This bot loads a trained MCCFR strategy and uses it to make decisions.
The strategy is learned through External Sampling MCCFR and represents
an approximation to Nash equilibrium.
"""

from collections import defaultdict
from skeleton.states import RoundState
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
import random
import pickle
from pathlib import Path


# Import our MCCFR utility functions (need to be standalone)

STARTING_STACK = 400
# ============================================================================
# CARD ABSTRACTION (SUIT ISOMORPHISM)
# ============================================================================


def get_card_rank(card_str):
    """Extract rank from card string like 'As' -> 14"""
    rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    return rank_map[card_str[0]]


def get_card_suit(card_str):
    """Extract suit from card string like 'As' -> 's'"""
    return card_str[1]


def canonicalize_cards(hand, board):
    """
    Map suits to canonical ordering based on rank hierarchy.
    Returns tuple of (canonical_hand, canonical_board).
    """
    all_cards = list(hand) + list(board)
    card_strings = [str(c) for c in all_cards]

    # Sort by rank (descending) to assign canonical suits
    sorted_cards = sorted(card_strings, key=get_card_rank, reverse=True)

    suit_map = {}
    next_canonical_suit = 0

    for card in sorted_cards:
        original_suit = get_card_suit(card)
        if original_suit not in suit_map:
            suit_map[original_suit] = next_canonical_suit
            next_canonical_suit += 1

    # Apply mapping to all cards
    def canonicalize(card_str):
        rank = card_str[0]
        suit = get_card_suit(card_str)
        canonical_suit = suit_map[suit]
        return f"{rank}s{canonical_suit}"

    canonical_hand = [canonicalize(str(c)) for c in hand]
    canonical_board = [canonicalize(str(c)) for c in board]

    return canonical_hand, canonical_board


# ============================================================================
# INFORMATION SET ENCODING
# ============================================================================

def get_infoset(state, player):
    """Encode game state into canonical information set string."""

    canonical_hand, canonical_board = canonicalize_cards(
        state.hands[player],
        state.board
    )

    # Sort cards within hand and board (order doesn't matter - Markov property)
    hand_str = ','.join(sorted(canonical_hand))
    board_str = ','.join(sorted(canonical_board)) if canonical_board else ''

    street = state.street

    # Extract betting history
    history = []
    current = state
    while current.previous_state is not None:
        prev = current.previous_state
        if hasattr(current, 'board') and hasattr(prev, 'board'):
            # Check for discard actions (board growth within same street)
            if len(current.board) > len(prev.board):
                if prev.street in (2, 3) and current.street == prev.street:
                    history.append('D')
            # Check for street transitions
            elif current.street != prev.street:
                if current.pips[0] == current.pips[1]:
                    history.append('X')
                else:
                    history.append('C')
            # Check for bets/raises
            elif current.pips != prev.pips:
                pot_from_previous_streets = (
                    2 * STARTING_STACK) - sum(prev.stacks)
                pot_this_street = sum(prev.pips)
                pot_before_bet = pot_from_previous_streets + pot_this_street

                bet_amount = max(current.pips) - max(prev.pips)

                if pot_before_bet > 0:
                    bet_to_pot_ratio = bet_amount / pot_before_bet
                    if bet_to_pot_ratio < 0.5:
                        history.append('r')
                    elif bet_to_pot_ratio < 1.0:
                        history.append('R')
                    else:
                        history.append('B')
                else:
                    history.append('R')
        current = prev

    history.reverse()
    history_str = ''.join(history[-20:])

    infoset = f"S{street}|H:{hand_str}|B:{board_str}|A:{history_str}"
    return infoset


# ============================================================================
# ACTION ENCODING
# ============================================================================

def action_to_key(action, state=None, active_player=None):
    """Convert an action object to a string key."""
    if isinstance(action, FoldAction):
        return "FOLD"
    elif isinstance(action, CallAction):
        return "CALL"
    elif isinstance(action, CheckAction):
        return "CHECK"
    elif isinstance(action, RaiseAction):
        return f"RAISE_{action.amount}"
    elif isinstance(action, DiscardAction):
        if state is not None and active_player is not None:
            canonical_hand, _ = canonicalize_cards(
                state.hands[active_player],
                state.board
            )
            canonical_card = canonical_hand[action.card]
            return f"DISCARD_{canonical_card}"
        else:
            return f"DISCARD_{action.card}"
    return str(action)


def get_legal_actions_list(state):
    """Get list of legal actions with concrete values."""
    legal_action_types = state.legal_actions()
    actions = []
    active = state.button % 2

    for action_type in legal_action_types:
        if action_type == RaiseAction:
            min_raise, max_raise = state.raise_bounds()
            raise_sizes = [
                min_raise,
                (min_raise + max_raise) // 2,
                max_raise
            ]
            raise_sizes = sorted(list(set(raise_sizes)))
            for size in raise_sizes:
                actions.append(RaiseAction(size))
        elif action_type == DiscardAction:
            for card_idx in range(len(state.hands[active])):
                actions.append(DiscardAction(card_idx))
        else:
            actions.append(action_type())

    return actions


# ============================================================================
# STRATEGY EXTRACTION
# ============================================================================

def get_average_strategy(infoset, legal_actions, strategy_table, state, player):
    """Extract average strategy from the strategy table."""
    strategy = {}
    action_keys = [action_to_key(a, state, player) for a in legal_actions]

    cumulative_strategy = strategy_table.get(infoset, {})
    total = sum(cumulative_strategy.get(key, 0.0) for key in action_keys)

    if total > 0:
        strategy = {key: cumulative_strategy.get(
            key, 0.0) / total for key in action_keys}
    else:
        # Uniform distribution if no strategy data
        uniform_prob = 1.0 / len(action_keys) if action_keys else 0.0
        strategy = {key: uniform_prob for key in action_keys}

    return strategy


# ============================================================================
# PLAYER BOT
# ============================================================================

class Player(Bot):
    """
    MCCFR Player Bot - Uses learned strategy from External Sampling MCCFR.
    """

    def __init__(self):
        """
        Initialize the bot by loading the trained strategy.
        """
        self.strategy_table = None
        self.regret_table = None
        self.use_mccfr = False

        # Try to load the strategy
        strategy_path = Path(__file__).parent / 'mccfr_checkpoint_1000.pkl'

        if strategy_path.exists():
            try:
                with open(strategy_path, 'rb') as f:
                    strategy_data = pickle.load(f)

                self.strategy_table = defaultdict(
                    lambda: defaultdict(float), strategy_data['strategy_table'])
                self.regret_table = defaultdict(
                    lambda: defaultdict(float), strategy_data['regret_table'])
                self.use_mccfr = True

                print(f"✓ MCCFR Strategy Loaded!")
                print(f"  Infosets: {len(self.strategy_table)}")
                print(f"  Ready to play with learned Nash approximation")
            except Exception as e:
                print(f"Failed to load strategy: {e}")
                print("Falling back to random play")
        else:
            print(f"Strategy file not found: {strategy_path}")
            print("Train MCCFR first and save with save_strategy('mccfr_strategy.pkl')")
            print("Falling back to random play")

    def handle_new_round(self, game_state, round_state, active):
        """
        Called when a new round starts.
        """
        pass  # MCCFR is stateless, nothing to do here

    def handle_round_over(self, game_state, terminal_state, active):
        """
        Called when a round ends.
        """
        pass  # MCCFR doesn't need to learn during play

    def get_action(self, game_state, round_state, active):
        """
        Query the learned strategy for the current game state.

        Args:
            game_state: The game state object
            round_state: The current round state
            active: The active player (0 or 1)

        Returns:
            Action to take
        """
        try:
            legal_actions = get_legal_actions_list(round_state)

            if not legal_actions:
                # Should never happen, but handle gracefully
                return CheckAction()

            if not self.use_mccfr or not self.strategy_table:
                # Fallback to random play
                return random.choice(legal_actions)

            # Get information set for current state
            infoset = get_infoset(round_state, active)

            # Query learned strategy
            strategy = get_average_strategy(
                infoset, legal_actions, self.strategy_table, round_state, active)

            # Sample action according to learned probabilities
            action_keys = [action_to_key(a, round_state, active)
                           for a in legal_actions]
            probs = [strategy[key] for key in action_keys]

            # Ensure probabilities sum to 1.0 (numerical stability)
            prob_sum = sum(probs)
            if prob_sum > 0:
                probs = [p / prob_sum for p in probs]
            else:
                # Uniform if no data
                probs = [1.0 / len(probs)] * len(probs)

            selected_action = random.choices(
                legal_actions, weights=probs, k=1)[0]

            return selected_action

        except Exception as e:
            # If anything fails, fall back to random play
            print(f"Error in get_action: {e}")
            import traceback
            traceback.print_exc()

            # Safe fallback
            legal_actions = round_state.legal_actions()
            if CheckAction in legal_actions:
                return CheckAction()
            elif CallAction in legal_actions:
                return CallAction()
            elif FoldAction in legal_actions:
                return FoldAction()
            else:
                return list(legal_actions)[0]()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
