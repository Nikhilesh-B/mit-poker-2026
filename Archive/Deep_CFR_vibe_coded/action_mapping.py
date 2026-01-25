"""
Action mapping for Deep CFR

Maps between discrete action indices (0-7) used by neural network
and actual game actions used by the engine.
"""

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from enum import IntEnum


class ActionIndex(IntEnum):
    """
    Discrete action space for neural network.
    Network outputs 8 values, one for each action.
    """
    FOLD = 0
    CHECK_CALL = 1
    RAISE_SMALL = 2
    RAISE_MED = 3
    RAISE_LARGE = 4
    DISCARD_0 = 5
    DISCARD_1 = 6
    DISCARD_2 = 7


def get_legal_mask(round_state, player_id):
    """
    Create a binary mask for legal actions.

    Args:
        round_state: RoundState object
        player_id: int (0 or 1)

    Returns:
        mask: torch.Tensor of shape [8] with 1.0 for legal, 0.0 for illegal

    Example:
        Preflop facing bet: [1, 1, 1, 1, 1, 0, 0, 0]
        (can fold, call, raise but not discard)
    """
    import torch

    mask = torch.zeros(8, dtype=torch.float32)
    legal_actions = round_state.legal_actions()

    # Map action types to indices
    if FoldAction in legal_actions:
        mask[ActionIndex.FOLD] = 1.0

    if CheckAction in legal_actions or CallAction in legal_actions:
        mask[ActionIndex.CHECK_CALL] = 1.0

    if RaiseAction in legal_actions:
        mask[ActionIndex.RAISE_SMALL] = 1.0
        mask[ActionIndex.RAISE_MED] = 1.0
        mask[ActionIndex.RAISE_LARGE] = 1.0

    # Discard actions - check which cards are available
    if DiscardAction in legal_actions:
        active = round_state.button % 2
        num_cards = len(round_state.hands[active])
        for i in range(min(3, num_cards)):
            mask[ActionIndex.DISCARD_0 + i] = 1.0

    return mask


def action_index_to_engine_action(action_idx, round_state):
    """
    Convert network action index to engine action object.

    Args:
        action_idx: int (0-7)
        round_state: RoundState object (needed for raise amounts)

    Returns:
        Action object (FoldAction, CallAction, etc.)
    """
    action_idx = int(action_idx)

    if action_idx == ActionIndex.FOLD:
        return FoldAction()

    elif action_idx == ActionIndex.CHECK_CALL:
        # Determine if it's check or call based on state
        active = round_state.button % 2
        continue_cost = round_state.pips[1-active] - round_state.pips[active]
        if continue_cost == 0:
            return CheckAction()
        else:
            return CallAction()

    elif action_idx in [ActionIndex.RAISE_SMALL, ActionIndex.RAISE_MED, ActionIndex.RAISE_LARGE]:
        # Get raise bounds
        min_raise, max_raise = round_state.raise_bounds()

        # Map to specific raise amount
        if action_idx == ActionIndex.RAISE_SMALL:
            amount = min_raise
        elif action_idx == ActionIndex.RAISE_LARGE:
            amount = max_raise
        else:  # RAISE_MED
            amount = int(round((min_raise + max_raise) / 2))
            amount = max(min_raise, min(max_raise, amount))  # Clamp

        return RaiseAction(amount)

    elif action_idx in [ActionIndex.DISCARD_0, ActionIndex.DISCARD_1, ActionIndex.DISCARD_2]:
        card_index = action_idx - ActionIndex.DISCARD_0
        return DiscardAction(card_index)

    else:
        raise ValueError(f"Invalid action index: {action_idx}")


# Action names for debugging
ACTION_NAMES = [
    "Fold",
    "Check/Call",
    "Raise Small",
    "Raise Medium",
    "Raise Large",
    "Discard Card 0",
    "Discard Card 1",
    "Discard Card 2"
]


if __name__ == "__main__":
    print("Action Mapping:")
    for i, name in enumerate(ACTION_NAMES):
        print(f"  {i}: {name}")
