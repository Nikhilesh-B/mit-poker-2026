"""
Action Mapping Utilities for Deep CFR Integration

This module handles the mapping between:
- Network output indices [0-18] → Action keys used in MCCFR
- Legal vs illegal actions (network outputs all 19, but only some are legal)

The network outputs 19 actions in this order:
0: discard0, 1: discard1, 2: discard2, 3: check, 4: call, 5: fold, 
6: raise_25_pot, 7: raise_50_pot, 8: raise_75_pot, 9: raise_100_pot,
10: raise_150_pot, 11: raise_200_pot, 12: raise_250_pot, 13: raise_300_pot,
14: raise_350_pot, 15: raise_400_pot, 16: raise_450_pot, 17: raise_500_pot,
18: raise_all_in

Pot-relative raises generalize across different game states without needing
to store raise bounds with each sample.
"""

import torch
from typing import Dict, List, Optional
from custom_engine import RoundState, RaiseAction
from config import STARTING_STACK


# Number of network outputs (actions)
NUM_ACTIONS = 19

# Pot-relative raise sizes (as fractions of pot)
# 25%, 50%, 75%, 100%, 150%, then 50% increments from 200% to 500%, plus all-in
POT_RAISE_FRACTIONS = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # 12 fractions + all-in

# Fixed mapping: Network output index → Action type name
# This is the order the network outputs actions
NETWORK_ACTION_TYPES = [
    'DISCARD_0',        # Index 0
    'DISCARD_1',        # Index 1
    'DISCARD_2',        # Index 2
    'CHECK',            # Index 3
    'CALL',             # Index 4
    'FOLD',             # Index 5
    'RAISE_25_POT',     # Index 6 - 25% of pot
    'RAISE_50_POT',     # Index 7 - 50% of pot
    'RAISE_75_POT',     # Index 8 - 75% of pot
    'RAISE_100_POT',    # Index 9 - 100% of pot (pot-sized)
    'RAISE_150_POT',    # Index 10 - 150% of pot
    'RAISE_200_POT',    # Index 11 - 200% of pot
    'RAISE_250_POT',    # Index 12 - 250% of pot
    'RAISE_300_POT',    # Index 13 - 300% of pot
    'RAISE_350_POT',    # Index 14 - 350% of pot
    'RAISE_400_POT',    # Index 15 - 400% of pot
    'RAISE_450_POT',    # Index 16 - 450% of pot
    'RAISE_500_POT',    # Index 17 - 500% of pot
    'RAISE_ALL_IN'      # Index 18 - All-in
]

# Map raise action names to network indices
RAISE_ACTION_TO_INDEX = {
    'RAISE_25_POT': 6,
    'RAISE_50_POT': 7,
    'RAISE_75_POT': 8,
    'RAISE_100_POT': 9,
    'RAISE_150_POT': 10,
    'RAISE_200_POT': 11,
    'RAISE_250_POT': 12,
    'RAISE_300_POT': 13,
    'RAISE_350_POT': 14,
    'RAISE_400_POT': 15,
    'RAISE_450_POT': 16,
    'RAISE_500_POT': 17,
    'RAISE_ALL_IN': 18
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


def calculate_pot_relative_raise(state: RoundState, pot_fraction: float) -> int:
    """
    Calculate a raise amount based on pot fraction.
    
    Args:
        state: Current RoundState
        pot_fraction: Fraction of pot to raise (e.g., 0.5 for half-pot)
        
    Returns:
        Raise amount clamped to legal bounds
    """
    pot = get_pot_size(state)
    min_raise, max_raise = state.raise_bounds()
    
    # Calculate raise amount as fraction of pot
    target_raise = int(pot * pot_fraction)
    
    # Clamp to legal bounds
    return max(min_raise, min(target_raise, max_raise))


def amount_to_pot_relative_key(amount: int, state: RoundState) -> str:
    """
    Convert a raise amount to a pot-relative action key.
    
    Args:
        amount: Raise amount in chips
        state: Current RoundState
        
    Returns:
        Action key like 'RAISE_50_POT' or 'RAISE_ALL_IN'
    """
    pot = get_pot_size(state)
    min_raise, max_raise = state.raise_bounds()
    
    # Check if this is an all-in
    if amount >= max_raise:
        return 'RAISE_ALL_IN'
    
    # Calculate pot fraction
    if pot > 0:
        pot_fraction = amount / pot
    else:
        pot_fraction = 1.0  # Default to pot-sized if pot is somehow 0
    
    # Map to closest bucket using midpoint boundaries
    # Buckets: 25%, 50%, 75%, 100%, 150%, 200%, 250%, 300%, 350%, 400%, 450%, 500%
    if pot_fraction <= 0.375:      # <= 37.5% -> 25%
        return 'RAISE_25_POT'
    elif pot_fraction <= 0.625:    # <= 62.5% -> 50%
        return 'RAISE_50_POT'
    elif pot_fraction <= 0.875:    # <= 87.5% -> 75%
        return 'RAISE_75_POT'
    elif pot_fraction <= 1.25:     # <= 125% -> 100%
        return 'RAISE_100_POT'
    elif pot_fraction <= 1.75:     # <= 175% -> 150%
        return 'RAISE_150_POT'
    elif pot_fraction <= 2.25:     # <= 225% -> 200%
        return 'RAISE_200_POT'
    elif pot_fraction <= 2.75:     # <= 275% -> 250%
        return 'RAISE_250_POT'
    elif pot_fraction <= 3.25:     # <= 325% -> 300%
        return 'RAISE_300_POT'
    elif pot_fraction <= 3.75:     # <= 375% -> 350%
        return 'RAISE_350_POT'
    elif pot_fraction <= 4.25:     # <= 425% -> 400%
        return 'RAISE_400_POT'
    elif pot_fraction <= 4.75:     # <= 475% -> 450%
        return 'RAISE_450_POT'
    else:                          # > 475% -> 500%
        return 'RAISE_500_POT'


def pot_relative_key_to_amount(key: str, state: RoundState) -> int:
    """
    Convert a pot-relative action key to an actual raise amount.
    
    Args:
        key: Action key like 'RAISE_50_POT'
        state: Current RoundState
        
    Returns:
        Raise amount in chips
    """
    pot = get_pot_size(state)
    min_raise, max_raise = state.raise_bounds()
    
    # Map key to pot fraction
    key_to_fraction = {
        'RAISE_25_POT': 0.25,
        'RAISE_50_POT': 0.50,
        'RAISE_75_POT': 0.75,
        'RAISE_100_POT': 1.0,
        'RAISE_150_POT': 1.5,
        'RAISE_200_POT': 2.0,
        'RAISE_250_POT': 2.5,
        'RAISE_300_POT': 3.0,
        'RAISE_350_POT': 3.5,
        'RAISE_400_POT': 4.0,
        'RAISE_450_POT': 4.5,
        'RAISE_500_POT': 5.0,
        'RAISE_ALL_IN': None  # Special case
    }
    
    if key == 'RAISE_ALL_IN':
        return max_raise
    
    fraction = key_to_fraction.get(key)
    if fraction is None:
        # Legacy key format - try to extract amount
        if key.startswith('RAISE_'):
            try:
                return int(key.split('_')[1])
            except (ValueError, IndexError):
                return min_raise
        return min_raise
    
    # Calculate amount and clamp to legal bounds
    target = int(pot * fraction)
    return max(min_raise, min(target, max_raise))


def get_all_network_action_keys(state: RoundState = None, active_player: int = None, 
                                mccfr_instance=None) -> List[str]:
    """
    Get all 13 action keys in network output order.
    
    This returns the fixed set of pot-relative action keys that the network outputs.
    No state is required since we use pot-relative naming.
    
    Args:
        state: Current RoundState (optional, not used for pot-relative keys)
        active_player: Player index (optional, not used)
        mccfr_instance: Optional MCCFR instance (not used)
        
    Returns:
        List of 13 action keys in network output order
    """
    return list(NETWORK_ACTION_TYPES)


def map_network_output_to_actions(network_output: torch.Tensor, 
                                  state: RoundState,
                                  active_player: int,
                                  mccfr_instance) -> Dict[str, float]:
    """
    Map network output tensor [13] to action_key -> regret dictionary.
    
    The network outputs regrets for all 13 actions. This function maps
    those to the action keys used by MCCFR.
    
    Strategy:
    - For discard actions: Map network indices to legal discard actions by position
    - For betting actions: Map directly (CHECK, CALL, FOLD)
    - For raises: Use pot-relative keys directly
    
    Args:
        network_output: Tensor of shape [13] with regrets for each action
        state: Current RoundState (needed for legal actions)
        active_player: Player index
        mccfr_instance: MCCFR instance (for getting legal actions and action_to_key)
        
    Returns:
        Dictionary mapping action_key -> regret value (only for legal actions)
    """
    # Get legal actions and their keys
    legal_actions = mccfr_instance.get_legal_actions_list(state)
    legal_keys = {mccfr_instance.action_to_key(a, state, active_player) for a in legal_actions}
    
    # Map network output to action keys
    regret_dict = {}
    
    # Map each network output index
    for i in range(NUM_ACTIONS):
        network_regret = network_output[i].item()
        action_key = NETWORK_ACTION_TYPES[i]
        
        if i < 3:
            # Discard actions (0, 1, 2) - using position-based keys
            if action_key in legal_keys:
                regret_dict[action_key] = network_regret
        elif i in (3, 4, 5):
            # CHECK, CALL, FOLD
            if action_key in legal_keys:
                regret_dict[action_key] = network_regret
        else:
            # Raise actions (6-12) - pot-relative keys
            if action_key in legal_keys:
                regret_dict[action_key] = network_regret
    
    return regret_dict


def apply_legal_action_mask(regret_dict: Dict[str, float],
                           legal_actions: List,
                           state: RoundState,
                           active_player: int,
                           mccfr_instance) -> Dict[str, float]:
    """
    Mask out illegal actions by setting their regrets to very negative values.
    
    The network outputs regrets for ALL 9 actions, but only some are legal.
    This function masks illegal actions so they don't affect regret matching.
    
    Args:
        regret_dict: Dictionary mapping action_key -> regret (from network)
        legal_actions: List of legal action instances
        state: Current RoundState
        active_player: Player index
        mccfr_instance: MCCFR instance (for action_to_key method)
        
    Returns:
        Dictionary with illegal actions masked (set to -1000.0)
    """
    # Get legal action keys
    legal_action_keys = {
        mccfr_instance.action_to_key(action, state, active_player)
        for action in legal_actions
    }
    
    # Create masked regrets
    masked_regrets = {}
    for action_key, regret in regret_dict.items():
        if action_key in legal_action_keys:
            # Legal action: keep original regret
            masked_regrets[action_key] = regret
        else:
            # Illegal action: mask with very negative regret
            masked_regrets[action_key] = -1000.0
    
    return masked_regrets


def fill_illegal_action_regrets(computed_regrets: Dict[str, float],
                                legal_actions: List,
                                state: RoundState,
                                active_player: int,
                                mccfr_instance) -> Dict[str, float]:
    """
    Fill in regrets for illegal actions with large negative value.
    
    When MCCFR computes regrets, it only computes them for legal actions.
    For training the network, we need regrets for ALL 9 actions.
    This function fills in illegal actions with -1000.0 so the network
    learns that illegal actions should have negative regrets.
    
    Args:
        computed_regrets: Dictionary with regrets for legal actions only (MCCFR keys)
        legal_actions: List of legal action instances
        state: Current RoundState
        active_player: Player index
        mccfr_instance: MCCFR instance (for action_to_key)
        
    Returns:
        Dictionary with regrets for ALL legal actions (keeps computed regrets)
        Note: We don't add illegal actions here - they'll be filled when converting to tensor
    """
    # Start with computed regrets (all legal actions)
    filled_regrets = computed_regrets.copy()
    
    # All computed regrets are for legal actions, so we're done
    # Illegal actions will be handled when converting to tensor format
    return filled_regrets


def regrets_dict_to_tensor(regrets_dict: Dict[str, float],
                           state: RoundState = None,
                           active_player: int = None,
                           mccfr_instance = None) -> torch.Tensor:
    """
    Convert regrets dictionary to tensor in network output order.
    
    This is the inverse of map_network_output_to_actions - it takes
    MCCFR action keys and converts them back to network output order.
    
    With pot-relative raises, we no longer need state to determine raise amounts.
    The regrets_dict already contains pot-relative keys.
    
    Args:
        regrets_dict: Dictionary mapping action_key -> regret (pot-relative keys)
        state: Current RoundState (optional, used for legacy key conversion)
        active_player: Player index (optional)
        mccfr_instance: MCCFR instance (optional)
        
    Returns:
        Tensor of shape [13] with regrets in network output order
    """
    # Build tensor in network output order [13]
    regrets_list = []
    
    for action_key in NETWORK_ACTION_TYPES:
        # Get regret for this action, default to 0.0 if not present
        regret = regrets_dict.get(action_key, 0.0)
        regrets_list.append(regret)
    
    return torch.tensor(regrets_list, dtype=torch.float32)
