"""
Action Mapping Utilities for Deep CFR Integration

This module handles the mapping between:
- Network output indices [0-8] → Action keys used in MCCFR
- Legal vs illegal actions (network outputs all 9, but only some are legal)

The network outputs 9 actions in this order:
0: discard0, 1: discard1, 2: discard2, 3: check, 4: call, 5: fold, 
6: raise_small, 7: raise_medium, 8: raise_large
"""

import torch
from typing import Dict, List, Optional
from custom_engine import RoundState, RaiseAction
from config import STARTING_STACK


# Fixed mapping: Network output index → Action type name
# This is the order the network outputs actions
NETWORK_ACTION_TYPES = [
    'DISCARD_0',      # Index 0
    'DISCARD_1',      # Index 1
    'DISCARD_2',      # Index 2
    'CHECK',          # Index 3
    'CALL',           # Index 4
    'FOLD',           # Index 5
    'RAISE_SMALL',    # Index 6
    'RAISE_MEDIUM',   # Index 7
    'RAISE_LARGE'     # Index 8
]


def get_all_network_action_keys(state: RoundState, active_player: int, 
                                mccfr_instance=None) -> List[str]:
    """
    Get all possible action keys that the network can output.
    
    This includes all 9 actions, but raise amounts depend on the current state.
    The network outputs "small/medium/large" raises which map to actual amounts.
    
    For discard actions, we use placeholder keys that will be mapped properly
    when we have the actual legal actions.
    
    Args:
        state: Current RoundState
        active_player: Player index (0 or 1)
        mccfr_instance: Optional MCCFR instance (for getting canonical discard keys)
        
    Returns:
        List of 9 action keys in network output order
    """
    # Get raise bounds for this state
    min_raise, max_raise = state.raise_bounds()
    mid_raise = (min_raise + max_raise) // 2
    
    # Build list of all 9 action keys
    action_keys = []
    
    # Discard actions (0, 1, 2)
    # Note: In MCCFR, discard keys use canonical card values (e.g., "DISCARD_13s0")
    # For network mapping, we use placeholder indices. The actual mapping happens
    # when we have legal actions and can use MCCFR's action_to_key method.
    # For now, we use generic keys that will be matched by type.
    for i in range(3):
        action_keys.append(f'DISCARD_PLACEHOLDER_{i}')
    
    # Check, Call, Fold (3, 4, 5)
    action_keys.extend(['CHECK', 'CALL', 'FOLD'])
    
    # Raises (6, 7, 8) - map to actual amounts based on state
    # Network outputs: RAISE_SMALL, RAISE_MEDIUM, RAISE_LARGE
    # MCCFR uses: RAISE_{amount} based on pot ratio
    # We'll map network raises to MCCFR keys based on raise_bounds
    action_keys.append(f'RAISE_{min_raise}')   # Small
    action_keys.append(f'RAISE_{mid_raise}')   # Medium  
    action_keys.append(f'RAISE_{max_raise}')   # Large
    
    return action_keys


def map_network_output_to_actions(network_output: torch.Tensor, 
                                  state: RoundState,
                                  active_player: int,
                                  mccfr_instance) -> Dict[str, float]:
    """
    Map network output tensor [9] to action_key -> regret dictionary.
    
    The network outputs regrets for all 9 actions. This function maps
    those to the action keys used by MCCFR.
    
    Strategy:
    - For discard actions: Map network indices to legal discard actions by position
    - For betting actions: Map directly (CHECK, CALL, FOLD)
    - For raises: Map network small/medium/large to actual raise amounts
    
    Args:
        network_output: Tensor of shape [9] with regrets for each action
        state: Current RoundState (needed for raise amounts and legal actions)
        active_player: Player index (needed for raise amounts)
        mccfr_instance: MCCFR instance (for getting legal actions and action_to_key)
        
    Returns:
        Dictionary mapping action_key -> regret value (only for actions that exist)
    """
    from custom_engine import DiscardAction
    
    # Get legal actions
    legal_actions = mccfr_instance.get_legal_actions_list(state)
    legal_keys = {mccfr_instance.action_to_key(a, state, active_player) for a in legal_actions}
    
    # Map network output to action keys
    regret_dict = {}
    
    # Get discard actions (they're position-based in network, but canonical in MCCFR)
    discard_actions = [a for a in legal_actions if isinstance(a, DiscardAction)]
    
    # Map each network output index
    for i in range(9):
        network_regret = network_output[i].item()
        
        if i < 3:
            # Discard actions (0, 1, 2)
            # Map network discard index to legal discard action by position
            if i < len(discard_actions):
                action_key = mccfr_instance.action_to_key(discard_actions[i], state, active_player)
                regret_dict[action_key] = network_regret
            # If no discard at this index, it's illegal - will be masked later
        elif i == 3:
            # Check
            if 'CHECK' in legal_keys:
                regret_dict['CHECK'] = network_regret
        elif i == 4:
            # Call
            if 'CALL' in legal_keys:
                regret_dict['CALL'] = network_regret
        elif i == 5:
            # Fold
            if 'FOLD' in legal_keys:
                regret_dict['FOLD'] = network_regret
        elif i == 6:
            # Raise small - find the smallest raise amount
            raise_keys = sorted([k for k in legal_keys if 'RAISE' in k])
            if raise_keys:
                # Use the first raise key (smallest)
                regret_dict[raise_keys[0]] = network_regret
        elif i == 7:
            # Raise medium - find the middle raise amount
            raise_keys = sorted([k for k in legal_keys if 'RAISE' in k])
            if len(raise_keys) >= 2:
                regret_dict[raise_keys[1]] = network_regret
            elif len(raise_keys) == 1:
                regret_dict[raise_keys[0]] = network_regret
        elif i == 8:
            # Raise large - find the largest raise amount
            raise_keys = sorted([k for k in legal_keys if 'RAISE' in k])
            if raise_keys:
                regret_dict[raise_keys[-1]] = network_regret
    
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
                           state: RoundState,
                           active_player: int,
                           mccfr_instance) -> torch.Tensor:
    """
    Convert regrets dictionary to tensor in network output order.
    
    This is the inverse of map_network_output_to_actions - it takes
    MCCFR action keys and converts them back to network output order.
    
    Args:
        regrets_dict: Dictionary mapping action_key -> regret (MCCFR keys)
        state: Current RoundState (for action key order)
        active_player: Player index (for action key order)
        mccfr_instance: MCCFR instance (for getting legal actions)
        
    Returns:
        Tensor of shape [9] with regrets in network output order
    """
    # Get legal actions to map discard actions properly
    legal_actions = mccfr_instance.get_legal_actions_list(state)
    
    # Build tensor in network output order [9]
    regrets_list = []
    
    # Discard actions (0, 1, 2)
    from custom_engine import DiscardAction
    discard_actions = [a for a in legal_actions if isinstance(a, DiscardAction)]
    for i in range(3):
        if i < len(discard_actions):
            action_key = mccfr_instance.action_to_key(discard_actions[i], state, active_player)
            regrets_list.append(regrets_dict.get(action_key, -1000.0))
        else:
            regrets_list.append(-1000.0)  # Illegal discard
    
    # Check, Call, Fold (3, 4, 5)
    regrets_list.append(regrets_dict.get('CHECK', -1000.0))
    regrets_list.append(regrets_dict.get('CALL', -1000.0))
    regrets_list.append(regrets_dict.get('FOLD', -1000.0))
    
    # Raises (6, 7, 8) - map based on raise amounts
    min_raise, max_raise = state.raise_bounds()
    mid_raise = (min_raise + max_raise) // 2
    
    # Find raise keys in regrets_dict (they might be RAISE_SMALL, RAISE_MEDIUM, RAISE_LARGE
    # or RAISE_{amount} depending on how they were computed)
    # We'll look for the actual raise amount keys
    regrets_list.append(regrets_dict.get(f'RAISE_{min_raise}', 
                                        regrets_dict.get('RAISE_SMALL', -1000.0)))
    regrets_list.append(regrets_dict.get(f'RAISE_{mid_raise}',
                                        regrets_dict.get('RAISE_MEDIUM', -1000.0)))
    regrets_list.append(regrets_dict.get(f'RAISE_{max_raise}',
                                        regrets_dict.get('RAISE_LARGE', -1000.0)))
    
    return torch.tensor(regrets_list, dtype=torch.float32)
