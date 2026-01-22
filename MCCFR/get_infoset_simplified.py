"""
Simplified get_infoset() function that uses the action_taken field
from custom_engine.py instead of inferring actions from state differences.

This is MUCH simpler and more reliable than the old approach!
"""

from typing import Union
from collections import namedtuple

# Assuming these are imported from your engine
# RoundState, TerminalState, FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction

STARTING_STACK = 400  # Adjust if different


def encode_action_for_history(action) -> str:
    """
    Convert an action object to a history string character.
    
    Returns:
        Single character representing the action:
        - 'D' for discard
        - 'X' for check
        - 'C' for call
        - 'F' for fold
        - 'r' for small raise (< 0.5 pot)
        - 'R' for medium raise (0.5-1.0 pot)
        - 'B' for big raise (> 1.0 pot)
        - '' (empty) for None/street transitions
    """
    if action is None:
        # Street transition (not a player action)
        return ''
    
    # Import action types (adjust imports as needed)
    from custom_engine import (FoldAction, CallAction, CheckAction, 
                                RaiseAction, DiscardAction)
    
    if isinstance(action, FoldAction):
        return 'F'
    elif isinstance(action, CallAction):
        return 'C'
    elif isinstance(action, CheckAction):
        return 'X'
    elif isinstance(action, DiscardAction):
        return 'D'
    elif isinstance(action, RaiseAction):
        # Calculate bet-to-pot ratio for bucketing
        # Note: We'd need the state to calculate pot size accurately
        # For now, return a simple encoding
        return 'R'  # Can refine this with pot info if needed
    else:
        return ''


def encode_raise_with_pot_info(action, state, prev_state) -> str:
    """
    Enhanced version that buckets raise sizes based on pot.
    
    Args:
        action: RaiseAction object
        state: Current state after the action
        prev_state: Previous state before the action
        
    Returns:
        'r' for small raise, 'R' for medium, 'B' for large
    """
    from custom_engine import RaiseAction
    
    if not isinstance(action, RaiseAction):
        return encode_action_for_history(action)
    
    # Calculate total pot
    pot_from_previous_streets = (2 * STARTING_STACK) - sum(prev_state.stacks)
    pot_this_street = sum(prev_state.pips)
    pot_before_bet = pot_from_previous_streets + pot_this_street
    
    # Calculate bet amount
    bet_amount = action.amount - max(prev_state.pips)
    
    if pot_before_bet > 0:
        bet_to_pot_ratio = bet_amount / pot_before_bet
        
        if bet_to_pot_ratio < 0.5:
            return 'r'  # Small raise
        elif bet_to_pot_ratio < 1.0:
            return 'R'  # Medium raise
        else:
            return 'B'  # Large raise
    else:
        return 'R'  # Default to medium if pot calculation fails


def get_infoset_simplified(state, player, canonicalize_cards_func) -> str:
    """
    SIMPLIFIED version of get_infoset using action_taken field.
    
    This is much cleaner than the old version that inferred actions!
    
    Args:
        state: Current RoundState
        player: Player index (0 or 1)
        canonicalize_cards_func: Function that takes (hand, board) and returns 
                                 (canonical_hand, canonical_board)
    
    Returns:
        Canonical information set string: "S{street}|H:{hand}|B:{board}|A:{actions}"
    """
    # 1. Canonicalize cards
    canonical_hand, canonical_board = canonicalize_cards_func(
        state.hands[player], 
        state.board
    )
    
    # 2. Sort for consistency (order doesn't matter - Markov property)
    hand_str = ','.join(sorted(canonical_hand))
    board_str = ','.join(sorted(canonical_board)) if canonical_board else ''
    
    # 3. Current street
    street = state.street
    
    # 4. Extract action history (THE EASY WAY!)
    history = []
    current = state
    prev = None
    
    while current.previous_state is not None:
        prev = current.previous_state
        
        # Simply read the action_taken field!
        if current.action_taken is not None:
            # For raises, we can use enhanced encoding with pot info
            if hasattr(current.action_taken, 'amount'):  # RaiseAction
                action_char = encode_raise_with_pot_info(
                    current.action_taken, current, prev)
            else:
                action_char = encode_action_for_history(current.action_taken)
            
            if action_char:  # Skip empty strings
                history.append(action_char)
        
        current = prev
    
    # Reverse to get chronological order
    history.reverse()
    history_str = ''.join(history[-20:])  # Keep last 20 actions
    
    # 5. Build final infoset string
    infoset = f"S{street}|H:{hand_str}|B:{board_str}|A:{history_str}"
    return infoset


# Example usage:
if __name__ == "__main__":
    print("Simplified get_infoset using action_taken field from custom_engine.py")
    print()
    print("Key improvements:")
    print("1. No more fragile inference logic")
    print("2. No confusion between discards and street transitions")
    print("3. Direct access to action history")
    print("4. Much faster (no complex state comparisons)")
    print()
    print("Usage in your MCCFR code:")
    print("  infoset = get_infoset_simplified(state, player, canonicalize_cards)")
