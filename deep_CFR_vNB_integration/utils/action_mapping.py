"""
Action Mapping Utilities for Deep CFR Integration

This module handles the mapping between:
- Network output indices [0-10] → Action keys used in MCCFR
- Legal vs illegal actions (network outputs all 11, but only some are legal)

The network outputs 11 actions in this order:
0: discard0, 1: discard1, 2: discard2, 3: check, 4: call, 5: fold, 
6: raise_tiny (<15), 7: raise_small (15-50), 8: raise_medium (50-125),
9: raise_large (125-250), 10: raise_allin (250+)

Uses absolute bet amounts since the game has fixed 400 starting stack.
"""

import torch
from typing import Dict, List, Optional
from custom_engine import RoundState, RaiseAction
from config import STARTING_STACK


# Number of network outputs (actions)
# 3 discards + 3 basic (check/call/fold) + 5 raise buckets = 11
NUM_ACTIONS = 11

# Absolute raise bucket boundaries (in chips)
# Buckets: <15, 15-50, 50-125, 125-250, 250+
RAISE_BUCKET_BOUNDARIES = [15, 50, 125, 250]  # 4 boundaries = 5 buckets

# Representative amounts for each bucket (used when generating raise actions)
# Midpoints for buckets 0-3, bucket 4 is all-in
RAISE_BUCKET_AMOUNTS = [8, 30, 85, 180]

# Fixed mapping: Network output index → Action type name
# This is the order the network outputs actions
NETWORK_ACTION_TYPES = [
    'DISCARD_0',        # Index 0
    'DISCARD_1',        # Index 1
    'DISCARD_2',        # Index 2
    'CHECK',            # Index 3
    'CALL',             # Index 4
    'FOLD',             # Index 5
    'RAISE_TINY',       # Index 6 - <15 chips
    'RAISE_SMALL',      # Index 7 - 15-50 chips
    'RAISE_MEDIUM',     # Index 8 - 50-125 chips
    'RAISE_LARGE',      # Index 9 - 125-250 chips
    'RAISE_ALL_IN'      # Index 10 - 250+ chips (effectively all-in)
]

# Map raise action names to network indices
RAISE_ACTION_TO_INDEX = {
    'RAISE_TINY': 6,
    'RAISE_SMALL': 7,
    'RAISE_MEDIUM': 8,
    'RAISE_LARGE': 9,
    'RAISE_ALL_IN': 10
}


def get_pot_size(state: RoundState) -> int:
    """
    Calculate the current pot size.

    Pot = (starting stacks * 2) - remaining stacks + current pips

    Args:
        state: Current RoundState

    Returns:
        Current pot size in chips
    """
    pot_from_previous_streets = (2 * STARTING_STACK) - sum(state.stacks)
    pot_this_street = sum(state.pips)
    return pot_from_previous_streets + pot_this_street


def calculate_raise_amount(state: RoundState, target_amount: int) -> int:
    """
    Calculate a raise amount clamped to legal bounds.

    Args:
        state: Current RoundState
        target_amount: Target raise amount in chips

    Returns:
        Raise amount clamped to legal bounds
    """
    min_raise, max_raise = state.raise_bounds()
    return max(min_raise, min(target_amount, max_raise))


def amount_to_raise_key(amount: int, state: RoundState) -> str:
    """
    Convert a raise amount to a bucket-based action key.

    Buckets (absolute amounts):
    - RAISE_TINY: < 15 chips
    - RAISE_SMALL: 15-50 chips
    - RAISE_MEDIUM: 50-125 chips
    - RAISE_LARGE: 125-250 chips
    - RAISE_ALL_IN: 250+ chips

    Args:
        amount: Raise amount in chips
        state: Current RoundState

    Returns:
        Action key like 'RAISE_MEDIUM' or 'RAISE_ALL_IN'
    """
    min_raise, max_raise = state.raise_bounds()

    # Check if this is an all-in (at or near max)
    if amount >= max_raise or amount >= 250:
        return 'RAISE_ALL_IN'

    # Map to bucket using boundaries: [15, 50, 125, 250]
    if amount < 15:
        return 'RAISE_TINY'
    elif amount < 50:
        return 'RAISE_SMALL'
    elif amount < 125:
        return 'RAISE_MEDIUM'
    elif amount < 250:
        return 'RAISE_LARGE'
    else:
        return 'RAISE_ALL_IN'


def raise_key_to_amount(key: str, state: RoundState) -> int:
    """
    Convert a raise action key to an actual raise amount.
    Uses representative amounts for each bucket.

    Args:
        key: Action key like 'RAISE_MEDIUM'
        state: Current RoundState

    Returns:
        Raise amount in chips
    """
    min_raise, max_raise = state.raise_bounds()

    # Map key to representative amount for each bucket
    key_to_amount = {
        'RAISE_TINY': 8,       # Midpoint of <15
        'RAISE_SMALL': 30,     # Midpoint of 15-50
        'RAISE_MEDIUM': 85,    # Midpoint of 50-125
        'RAISE_LARGE': 180,    # Midpoint of 125-250
        'RAISE_ALL_IN': None   # Special case
    }

    if key == 'RAISE_ALL_IN':
        return max_raise

    target = key_to_amount.get(key)
    if target is None:
        return min_raise

    # Clamp to legal bounds
    return max(min_raise, min(target, max_raise))


def get_all_network_action_keys(state: RoundState = None, active_player: int = None,
                                mccfr_instance=None) -> List[str]:
    """
    Get all 11 action keys in network output order.

    Args:
        state: Current RoundState (optional, not used)
        active_player: Player index (optional, not used)
        mccfr_instance: Optional MCCFR instance (not used)

    Returns:
        List of 11 action keys in network output order
    """
    return list(NETWORK_ACTION_TYPES)


def map_network_output_to_actions(network_output: torch.Tensor,
                                  state: RoundState,
                                  active_player: int,
                                  mccfr_instance) -> Dict[str, float]:
    """
    Map network output tensor [11] to action_key -> regret dictionary.

    Args:
        network_output: Tensor of shape [11] with regrets for each action
        state: Current RoundState (needed for legal actions)
        active_player: Player index
        mccfr_instance: MCCFR instance (for getting legal actions and action_to_key)

    Returns:
        Dictionary mapping action_key -> regret value (only for legal actions)
    """
    # Get legal actions and their keys
    legal_actions = mccfr_instance.get_legal_actions_list(state)
    legal_keys = {mccfr_instance.action_to_key(
        a, state, active_player) for a in legal_actions}

    # Map network output to action keys
    regret_dict = {}

    for i in range(NUM_ACTIONS):
        network_regret = network_output[i].item()
        action_key = NETWORK_ACTION_TYPES[i]

        if action_key in legal_keys:
            regret_dict[action_key] = network_regret

    return regret_dict


def apply_legal_action_mask(regret_dict: Dict[str, float],
                            legal_actions: List,
                            state: RoundState,
                            active_player: int,
                            mccfr_instance) -> Dict[str, float]:
    """
    Mask out illegal actions by setting their regrets to zero.

    This prevents gradients from flowing through illegal actions while keeping
    the values at zero (not very negative) so they don't affect softmax calculations.

    Args:
        regret_dict: Dictionary mapping action_key -> regret (from network)
        legal_actions: List of legal action instances
        state: Current RoundState
        active_player: Player index
        mccfr_instance: MCCFR instance (for action_to_key method)

    Returns:
        Dictionary with illegal actions masked (set to 0.0)
    """
    legal_action_keys = {
        mccfr_instance.action_to_key(action, state, active_player)
        for action in legal_actions
    }

    masked_regrets = {}
    for action_key, regret in regret_dict.items():
        if action_key in legal_action_keys:
            masked_regrets[action_key] = regret
        else:
            masked_regrets[action_key] = 0.0  # Zero out illegal actions

    return masked_regrets


def fill_illegal_action_regrets(computed_regrets: Dict[str, float],
                                legal_actions: List,
                                state: RoundState,
                                active_player: int,
                                mccfr_instance) -> Dict[str, float]:
    """
    Fill in regrets for illegal actions with large negative value.
    """
    return computed_regrets.copy()


def regrets_dict_to_tensor(regrets_dict: Dict[str, float],
                           state: RoundState = None,
                           active_player: int = None,
                           mccfr_instance=None) -> torch.Tensor:
    """
    Convert regrets dictionary to tensor in network output order.

    Args:
        regrets_dict: Dictionary mapping action_key -> regret

    Returns:
        Tensor of shape [11] with regrets in network output order
    """
    regrets_list = []

    for action_key in NETWORK_ACTION_TYPES:
        regret = regrets_dict.get(action_key, 0.0)
        regrets_list.append(regret)

    return torch.tensor(regrets_list, dtype=torch.float32)
