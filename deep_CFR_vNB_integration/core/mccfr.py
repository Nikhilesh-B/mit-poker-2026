"""
MCCFR (Monte Carlo Counterfactual Regret Minimization) Implementation

This is a clean, object-oriented implementation of MCCFR that can be easily
integrated with Deep CFR networks.

Usage:
    mccfr = MCCFR()
    mccfr.train(num_iterations=1000)
    action = mccfr.select_action(state, player=0)
    
Note:
    This module supports two modes of operation:
    1. Training mode: Uses custom_engine for MCCFR traversals with action_taken tracking
    2. Live play mode: Uses skeleton/states from the game engine without action_taken
    
    Action types are compared by name (not identity) to support both modes.
"""

import sys
import os

# Import from local custom_engine for training
from custom_engine import (
    RoundState as TrainingRoundState,
    TerminalState as TrainingTerminalState,
    FoldAction as TrainingFoldAction,
    CallAction as TrainingCallAction,
    CheckAction as TrainingCheckAction,
    RaiseAction as TrainingRaiseAction,
    DiscardAction as TrainingDiscardAction,
)
from config import STARTING_STACK, BIG_BLIND, SMALL_BLIND
from pkrbot import Deck
from utils.canon_cards import canon_cards

# Also import skeleton actions for live play
from skeleton.actions import (
    FoldAction as SkeletonFoldAction,
    CallAction as SkeletonCallAction,
    CheckAction as SkeletonCheckAction,
    RaiseAction as SkeletonRaiseAction,
    DiscardAction as SkeletonDiscardAction
)

import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional


# Helper functions to identify action types regardless of their source module
def is_fold_action(action_or_type):
    """Check if action or type is a FoldAction."""
    name = action_or_type.__class__.__name__ if not isinstance(
        action_or_type, type) else action_or_type.__name__
    return name == 'FoldAction'


def is_call_action(action_or_type):
    """Check if action or type is a CallAction."""
    name = action_or_type.__class__.__name__ if not isinstance(
        action_or_type, type) else action_or_type.__name__
    return name == 'CallAction'


def is_check_action(action_or_type):
    """Check if action or type is a CheckAction."""
    name = action_or_type.__class__.__name__ if not isinstance(
        action_or_type, type) else action_or_type.__name__
    return name == 'CheckAction'


def is_raise_action(action_or_type):
    """Check if action or type is a RaiseAction."""
    name = action_or_type.__class__.__name__ if not isinstance(
        action_or_type, type) else action_or_type.__name__
    return name == 'RaiseAction'


def is_discard_action(action_or_type):
    """Check if action or type is a DiscardAction."""
    name = action_or_type.__class__.__name__ if not isinstance(
        action_or_type, type) else action_or_type.__name__
    return name == 'DiscardAction'


def is_terminal_state(state):
    """Check if state is a TerminalState."""
    return state.__class__.__name__ == 'TerminalState'


def is_round_state(state):
    """Check if state is a RoundState."""
    return state.__class__.__name__ == 'RoundState'


# Aliases for backwards compatibility with existing code that uses training actions
RoundState = TrainingRoundState
TerminalState = TrainingTerminalState
FoldAction = TrainingFoldAction
CallAction = TrainingCallAction
CheckAction = TrainingCheckAction
RaiseAction = TrainingRaiseAction
DiscardAction = TrainingDiscardAction


class MCCFR:
    """
    Monte Carlo Counterfactual Regret Minimization (MCCFR) Algorithm

    This class implements the Deep CFR paper's external sampling approach:
    - Stores INSTANTANEOUS regrets (not cumulative) with iteration numbers
    - Collects strategy samples at opponent nodes for the strategy network

    Following the paper:
    - At traverser's nodes: store (infoset, iteration, instantaneous_regrets) in MV,p
    - At opponent's nodes: store (infoset, iteration, current_strategy) in MΠ
    """

    def __init__(self):
        """Initialize MCCFR with empty regret and strategy tables."""
        # Legacy cumulative regret table (for backward compatibility and regret matching)
        # r_I[a] = cumulative regret for action a at infoset I
        self.regret_table = defaultdict(lambda: defaultdict(float))

        # Strategy table: s_I[a] = cumulative strategy for action a at infoset I
        self.strategy_table = defaultdict(lambda: defaultdict(float))

        # Cache for infoset lookups (avoids recomputation)
        self.infoset_cache = {}

        # === Deep CFR Paper's Data Collection ===
        # Advantage memories MV,p: List of (infoset, iteration, instantaneous_regrets, player)
        self.advantage_memory = {
            0: [],  # Player 0's advantage samples
            1: []   # Player 1's advantage samples
        }

        # Strategy memory MΠ: List of (infoset, iteration, strategy, player)
        self.strategy_memory = []

        # Current CFR iteration (for linear weighting)
        self.current_iteration = 0

    def get_infoset(self, state: RoundState, player: int) -> str:
        """
        Get information set string for a state.

        Information set = what the player knows at a decision point:
        - Street (game phase)
        - Their hand (canonical)
        - Board cards (canonical)
        - Action history

        This method supports two modes:
        1. Training mode (custom_engine.py): Uses state.action_taken for efficient history
        2. Live play mode (skeleton/states.py): Reconstructs history from state differences

        Args:
            state: Current RoundState
            player: Player index (0 or 1)

        Returns:
            Infoset string: "S{street}|H:{hand_str}|B:{board_str}|A:{history_str}"
        """
        # Use canon_cards for canonical representation
        cards = canon_cards(state.hands[player], state.board)
        hand_str = cards.get_hand_str()
        board_str = cards.get_board_str() if state.board else ''
        street = state.street

        # Extract action history - supports both training and live play states
        # CRITICAL FIX: Add cycle detection to prevent infinite loops
        history = []
        current = state
        visited = set()  # Track visited states to detect cycles
        max_depth = 1000  # Safety limit for history reconstruction

        depth = 0
        while current.previous_state is not None and depth < max_depth:
            # Cycle detection: if we've seen this state before, we have a cycle
            state_id = id(current)
            if state_id in visited:
                raise RuntimeError(
                    f"Cycle detected in previous_state chain at depth {depth}! "
                    f"This indicates a bug in state.proceed() or state construction. "
                    f"State: street={current.street}, button={current.button}, "
                    f"pips={current.pips}"
                )
            visited.add(state_id)

            prev = current.previous_state

            # Method 1: Use action_taken if available (training with custom_engine)
            if hasattr(current, 'action_taken') and current.action_taken is not None:
                action = current.action_taken

                if is_fold_action(action):
                    history.append('F')
                elif is_call_action(action):
                    history.append('C')
                elif is_check_action(action):
                    history.append('X')
                elif is_discard_action(action):
                    history.append('D')
                elif is_raise_action(action):
                    # Use ABSOLUTE AMOUNT buckets (matching network output: 5 raise buckets)
                    bet_amount = max(current.pips) - max(prev.pips)

                    # Check for all-in first
                    if hasattr(prev, 'raise_bounds'):
                        _, max_raise = prev.raise_bounds()
                        if bet_amount >= max_raise or bet_amount >= 250:
                            history.append('Z')  # RAISE_ALL_IN
                            current = prev
                            depth += 1
                            continue

                    # Map to absolute amount bucket character
                    history.append(self._absolute_amount_to_char(bet_amount))

            # Method 2: Reconstruct action from state differences (live play with skeleton/states)
            else:
                action_key = self._infer_action_from_state_diff(current, prev)
                if action_key:
                    history.append(action_key)

            current = prev
            depth += 1

        if depth >= max_depth:
            raise RuntimeError(
                f"History reconstruction exceeded max_depth={max_depth}. "
                f"This suggests an extremely deep game tree or a bug."
            )

        # Reverse to get chronological order
        history.reverse()
        history_str = ''.join(history[-20:])  # Keep last 20 actions

        # Combine into canonical infoset string
        infoset = f"S{street}|H:{hand_str}|B:{board_str}|A:{history_str}"
        return infoset

    def _absolute_amount_to_char(self, amount: int) -> str:
        """
        Convert absolute raise amount to single character for action history encoding.

        Uses 5 absolute amount buckets matching the network output:
        '1' = RAISE_TINY (<15), '2' = RAISE_SMALL (15-50), '3' = RAISE_MEDIUM (50-125),
        '4' = RAISE_LARGE (125-250), 'Z' = RAISE_ALL_IN (250+)

        Bucket boundaries match action_to_key and action_mapping.py:
        - < 15 -> '1' (RAISE_TINY)
        - 15-50 -> '2' (RAISE_SMALL)
        - 50-125 -> '3' (RAISE_MEDIUM)
        - 125-250 -> '4' (RAISE_LARGE)
        - 250+ -> 'Z' (RAISE_ALL_IN, handled separately in caller)
        """
        if amount < 15:
            return '1'  # RAISE_TINY
        elif amount < 50:
            return '2'  # RAISE_SMALL
        elif amount < 125:
            return '3'  # RAISE_MEDIUM
        elif amount < 250:
            return '4'  # RAISE_LARGE
        else:
            return 'Z'  # RAISE_ALL_IN (should be caught earlier, but fallback)

    def _infer_action_from_state_diff(self, current, prev) -> Optional[str]:
        """
        Infer what action was taken by comparing current and previous states.

        This is used during live play when the RoundState doesn't have action_taken.

        Args:
            current: Current state
            prev: Previous state

        Returns:
            Single character action key ('F', 'C', 'X', 'D', '1'-'9', 'T', 'E', 'W', 'Z') or None
        """
        # Check for discard actions (board grows within same street during discard phase)
        if hasattr(current, 'board') and hasattr(prev, 'board'):
            if len(current.board) > len(prev.board):
                # Board grew - this is a discard action (streets 2 and 3 are discard streets)
                if prev.street in (2, 3) and current.street == prev.street:
                    return 'D'

        # Check for street transitions (Call or Check to end betting round)
        if current.street != prev.street:
            # Street changed - someone either called or checked to end the round
            # If pips were equal at end of betting, it was a check sequence
            # If pips were unequal, it was a call to end the street
            if prev.pips[0] == prev.pips[1]:
                return 'X'  # Check to end street
            else:
                return 'C'  # Call to end street

        # Check for bets/raises/calls/checks within same street
        if hasattr(current, 'pips') and hasattr(prev, 'pips'):
            if current.pips != prev.pips:
                # Pips changed within the same street
                # Determine if it's a call, bet, or raise

                # Calculate continue_cost from previous state to determine if we were facing a bet
                prev_continue_cost = prev.pips[1 -
                                               (prev.button % 2)] - prev.pips[prev.button % 2]

                # A call makes pips equal when facing a bet (continue_cost > 0)
                # Call: pips were unequal (facing bet), now equal, max pip unchanged
                if prev_continue_cost > 0 and current.pips[0] == current.pips[1] and max(current.pips) == max(prev.pips):
                    # Call (facing bet, pips equalized, max unchanged)
                    return 'C'

                # A bet/raise increases the max pip
                bet_amount = max(current.pips) - max(prev.pips)

                if bet_amount > 0:
                    # This is a bet or raise - check for all-in first
                    if hasattr(prev, 'raise_bounds'):
                        _, max_raise = prev.raise_bounds()
                        if bet_amount >= max_raise or bet_amount >= 250:
                            return 'Z'  # RAISE_ALL_IN

                    # Use absolute amount bucket
                    return self._absolute_amount_to_char(bet_amount)
                else:
                    # Bet amount is 0 or negative - this shouldn't happen, but if it does,
                    # check if we were facing a bet to determine call vs check
                    if prev_continue_cost > 0:
                        return 'C'  # Call when facing bet
                    else:
                        # Check when not facing bet (shouldn't happen if pips changed)
                        return 'X'

        # Check for checks (button advanced but pips didn't change)
        # This happens when continue_cost == 0 (no bet to call)
        if hasattr(current, 'button') and hasattr(prev, 'button'):
            if current.button != prev.button and current.pips == prev.pips:
                # Verify we weren't facing a bet
                prev_continue_cost = prev.pips[1 -
                                               (prev.button % 2)] - prev.pips[prev.button % 2]
                if prev_continue_cost == 0:
                    # Check (button moved, pips same, not facing bet)
                    return 'X'
                else:
                    # This shouldn't happen - if facing bet and pips unchanged, should be a call
                    # But pips are same, so it's actually a check (maybe all-in situation?)
                    return 'X'

        # Check for fold - this would be in terminal state, but handle just in case
        # Folds are usually detected by going to terminal state, so this is rare

        return None  # Couldn't determine action

    def action_to_key(self, action, state=None,
                      active_player: Optional[int] = None) -> str:
        """
        Convert an action object to a string key.

        This method works with action instances from both custom_engine and 
        skeleton.actions modules by checking type names instead of identity.

        For raises, uses ABSOLUTE AMOUNT bucket keys (matching network output):
        - RAISE_TINY: < 15 chips
        - RAISE_SMALL: 15-50 chips
        - RAISE_MEDIUM: 50-125 chips
        - RAISE_LARGE: 125-250 chips
        - RAISE_ALL_IN: 250+ chips or actual all-in

        Args:
            action: Action instance (FoldAction, CallAction, etc.)
            state: Current RoundState (required for RaiseAction pot calculation)
            active_player: Player index (optional)

        Returns:
            String key representing the action
        """
        # Handle action instances (check by name for cross-module compatibility)
        if is_fold_action(action):
            return "FOLD"
        elif is_call_action(action):
            return "CALL"
        elif is_check_action(action):
            return "CHECK"
        elif is_raise_action(action):
            # Use ABSOLUTE AMOUNT bucket keys for raises
            # Buckets: <15, 15-50, 50-125, 125-250, 250+
            if state is not None and hasattr(state, 'raise_bounds'):
                min_raise, max_raise = state.raise_bounds()
                amount = action.amount

                # Check for all-in first
                if amount >= max_raise or amount >= 250:
                    return "RAISE_ALL_IN"

                # Map to absolute amount bucket
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
            else:
                # Fallback without state - use all-in as default
                return "RAISE_ALL_IN"
        elif is_discard_action(action):
            # Use position-based keys (0, 1, 2) for network compatibility
            # The network outputs DISCARD_0, DISCARD_1, DISCARD_2 for positions in hand
            # action.card is already the position index (0, 1, or 2)
            return f"DISCARD_{action.card}"
        elif isinstance(action, type):
            # Handle action types (not instances) - check by name
            action_name = action.__name__
            if action_name == 'FoldAction':
                return "FOLD"
            elif action_name == 'CallAction':
                return "CALL"
            elif action_name == 'CheckAction':
                return "CHECK"
            elif action_name == 'RaiseAction':
                return "RAISE"
            elif action_name == 'DiscardAction':
                return "DISCARD"
        return str(action)

    def count_raises_this_round(self, state) -> int:
        """
        Count the number of raises in the current betting round.

        Traverses the state's previous_state chain to count raise actions
        on the current street.
        """
        count = 0
        current_street = state.street
        current = state

        while hasattr(current, 'previous_state') and current.previous_state is not None:
            prev = current.previous_state
            # Stop if we've gone back to a different street
            if hasattr(prev, 'street') and prev.street != current_street:
                break
            # Check if the action that led to current was a raise
            if hasattr(current, 'action_taken') and current.action_taken is not None:
                if is_raise_action(current.action_taken):
                    count += 1
            current = prev

        return count

    def get_legal_actions_list(self, state, use_skeleton_actions: bool = None) -> List:
        """
        Get list of legal actions with concrete values.

        Action abstraction:
        - Raises: 5 absolute amount buckets (<15, 15-50, 50-125, 125-250, 250+ chips)
        - Discards: All cards in hand
        - Check/Call/Fold: Single action each

        This method works with states from both custom_engine and skeleton/states.
        It automatically detects which action types to return based on the state type.

        Raises are capped at MAX_RAISES_PER_ROUND per betting round to limit tree depth.

        Args:
            state: Current RoundState (from either custom_engine or skeleton)
            use_skeleton_actions: If True, return skeleton action types. If False,
                return training action types. If None (default), auto-detect based
                on whether state has 'deck' attribute (training states have deck).

        Returns:
            List of concrete action instances

        Note: This method relies on state.legal_actions() which enforces poker rules:
        - CALL is only legal when continue_cost > 0 (facing a bet)
        - CHECK is only legal when continue_cost == 0 (not facing a bet)
        - At discard streets (2, 3): only DISCARD or CHECK are legal
        """
        legal_action_types = state.legal_actions()
        actions = []
        active = state.button % 2

        # Validate that legal_actions() is correctly enforcing poker rules
        # (This is a sanity check - state.legal_actions() should already enforce these rules)
        continue_cost = state.pips[1 - active] - state.pips[active]

        # Check if CALL and CHECK are in legal actions using helper functions
        has_call = any(is_call_action(t) for t in legal_action_types)
        has_check = any(is_check_action(t) for t in legal_action_types)

        # Verify: CALL and CHECK should never both be legal (except at discard streets where CHECK is special)
        if has_call and has_check:
            if state.street in (2, 3):
                # At discard streets, CALL should never be legal
                raise ValueError(
                    f"Invalid legal_actions: CALL should not be legal at discard street {state.street}")
            else:
                # At betting streets, both should never be legal
                raise ValueError(
                    f"Invalid legal_actions: Both CALL and CHECK legal at street {state.street}, continue_cost={continue_cost}")

        # Verify: CALL should only be legal when facing a bet (continue_cost > 0)
        if has_call and continue_cost <= 0:
            raise ValueError(
                f"Invalid legal_actions: CALL legal but continue_cost={continue_cost} <= 0 at street {state.street}")

        # Verify: CHECK should only be legal when not facing a bet (continue_cost == 0) or at discard streets
        if has_check and continue_cost > 0 and state.street not in (2, 3):
            raise ValueError(
                f"Invalid legal_actions: CHECK legal but continue_cost={continue_cost} > 0 at street {state.street}")

        # Check if raises are capped
        raises_capped = self.count_raises_this_round(
            state) >= MCCFR.MAX_RAISES_PER_ROUND

        # Auto-detect mode: training states have 'deck' attribute, skeleton states don't
        if use_skeleton_actions is None:
            # Training states (custom_engine) have 'deck' attribute
            use_skeleton_actions = not hasattr(state, 'deck')

        # Select action constructors based on mode
        if use_skeleton_actions:
            fold_cls = SkeletonFoldAction
            call_cls = SkeletonCallAction
            check_cls = SkeletonCheckAction
            raise_cls = SkeletonRaiseAction
            discard_cls = SkeletonDiscardAction
        else:
            fold_cls = TrainingFoldAction
            call_cls = TrainingCallAction
            check_cls = TrainingCheckAction
            raise_cls = TrainingRaiseAction
            discard_cls = TrainingDiscardAction

        for action_type in legal_action_types:
            action_name = action_type.__name__

            if action_name == 'RaiseAction':
                # Skip raises if we've hit the per-round cap
                if raises_capped:
                    continue

                # Discretize raise space into 5 absolute amount buckets
                # Buckets: <15, 15-50, 50-125, 125-250, 250+
                min_raise, max_raise = state.raise_bounds()

                # Representative amounts for each bucket (midpoints)
                # 8, 30, 85, 180, then all-in
                target_amounts = [8, 30, 85, 180]

                raise_sizes = set()
                for target in target_amounts:
                    # Clamp to legal bounds
                    clamped = max(min_raise, min(target, max_raise))
                    raise_sizes.add(clamped)

                # Always add all-in (max_raise)
                raise_sizes.add(max_raise)

                # Sort and create actions
                for size in sorted(raise_sizes):
                    actions.append(raise_cls(size))

            elif action_name == 'DiscardAction':
                # All cards in hand can be discarded
                for card_idx in range(len(state.hands[active])):
                    actions.append(discard_cls(card_idx))
            elif action_name == 'FoldAction':
                actions.append(fold_cls())
            elif action_name == 'CallAction':
                actions.append(call_cls())
            elif action_name == 'CheckAction':
                actions.append(check_cls())
            else:
                # Unknown action type - try to instantiate it
                try:
                    actions.append(action_type())
                except Exception:
                    pass

        return actions

    def regret_matching(self, regrets: Dict[str, float], actions: List,
                        state: Optional[RoundState] = None,
                        active_player: Optional[int] = None) -> Dict[str, float]:
        """
        Convert regrets to strategy using regret matching.

        Algorithm (following Deep CFR paper):
        1. Take max(regret, 0) for each action (positive regrets only)
        2. Sum all positive regrets
        3. If sum > 0: normalize to get probabilities
        4. If sum = 0 (all regrets negative): play HIGHEST-regret action with prob 1
           (Paper Figure 4: This reduces exploitability by ~50% vs uniform)

        Args:
            regrets: Dictionary mapping action_key -> cumulative regret
            actions: List of legal action instances
            state: Current RoundState (for action encoding)
            active_player: Player index (for action encoding)

        Returns:
            Dictionary mapping action_key -> probability
        """
        strategy = {}
        action_keys = [self.action_to_key(
            a, state, active_player) for a in actions]

        # Step 1: Clamp to positive regrets only
        positive_regrets = {key: max(regrets.get(key, 0.0), 0.0)
                            for key in action_keys}

        # Step 2: Sum positive regrets
        sum_positive = sum(positive_regrets.values())

        # Step 3 & 4: Normalize or handle all-negative regrets
        if sum_positive > 0:
            # Normal case: normalize positive regrets
            strategy = {
                key: positive_regrets[key] / sum_positive for key in action_keys}
        else:
            # All regrets negative: pick action with HIGHEST regret (deterministically)
            #
            # Paper Figure 4: "if the algorithm plays a uniform strategy when all
            # regrets are negative (i.e. standard regret matching), rather than the
            # highest-regret action, the final exploitability is also 50% higher."
            #
            # This deterministic choice reduces exploitability by ~50% vs uniform/softmax.
            if action_keys:
                regret_values = [regrets.get(k, 0.0) for k in action_keys]
                max_regret = max(regret_values)
                # Find the KEY with the highest (least negative) regret
                max_idx = regret_values.index(max_regret)
                best_key = action_keys[max_idx]
                # Assign probability 1 to that key, 0 to others
                # Note: Using key comparison (not index) handles duplicate keys correctly
                strategy = {key: 1.0 if key == best_key else 0.0
                            for key in action_keys}
            else:
                strategy = {}

        return strategy

    # Maximum traversal depth to prevent infinite game trees
    # With absolute amount raises and MAX_RAISES_PER_ROUND cap, betting is bounded
    # In MCCFR external sampling, we only explore all actions at traversing player nodes;
    # at opponent nodes we sample one path, so tree size is much smaller than full exploration
    # Depth 30 should cover full game (6 streets with multiple betting rounds)
    MAX_TRAVERSAL_DEPTH = 30

    # Maximum raises per betting round
    # This dramatically reduces tree size by capping the betting depth
    MAX_RAISES_PER_ROUND = 4

    def external_sampling(self, state, traversing_player: int,
                          collect_deep_cfr_samples: bool = True, _depth: int = 0) -> float:
        """
        External sampling MCCFR traversal (Deep CFR version).

        Following the paper's Algorithm 2:
        - Terminal states: return utility
        - Traversing player's nodes: compute INSTANTANEOUS regrets, store in MV,p
        - Opponent's nodes: sample action, store current strategy in MΠ

        Args:
            state: Current game state (RoundState or TerminalState)
            traversing_player: Player whose regrets we're updating (0 or 1)
            collect_deep_cfr_samples: If True, collect samples for Deep CFR training

        Returns:
            Utility value for traversing player at this node
        """
        # Depth limit: truncate extremely deep game trees
        # Return 0 (break-even estimate) when tree is too deep
        if _depth >= MCCFR.MAX_TRAVERSAL_DEPTH:
            return 0.0

        # Terminal state: return utility (check by name for cross-module compatibility)
        if is_terminal_state(state):
            return float(state.deltas[traversing_player])

        # Get information set
        active_player = state.button % 2
        state_id = id(state)

        if state_id in self.infoset_cache:
            infoset = self.infoset_cache[state_id]
        else:
            infoset = self.get_infoset(state, active_player)
            self.infoset_cache[state_id] = infoset

        # Get legal actions
        legal_actions = self.get_legal_actions_list(state)
        if not legal_actions:
            return 0.0

        # Get current strategy from regrets via regret matching
        strategy = self.regret_matching(
            self.regret_table[infoset],
            legal_actions,
            state,
            active_player
        )

        # Check if it's the traversing player's turn
        if active_player == traversing_player:
            # Traversing player: compute regrets for all actions
            action_values = {}
            node_value = 0.0

            # Compute value of each action
            for action in legal_actions:
                action_key = self.action_to_key(action, state, active_player)
                next_state = state.proceed(action)
                action_values[action_key] = self.external_sampling(
                    next_state, traversing_player, collect_deep_cfr_samples, _depth + 1
                )
                node_value += strategy[action_key] * action_values[action_key]

            # Compute INSTANTANEOUS regrets (not cumulative!)
            # Paper: r̃_t(I,a) = v(a) - Σ σ(a')·v(a')
            instantaneous_regrets = {}
            for action in legal_actions:
                action_key = self.action_to_key(action, state, active_player)
                regret = action_values[action_key] - node_value
                instantaneous_regrets[action_key] = regret

                # Also update cumulative table for regret matching (backward compatibility)
                self.regret_table[infoset][action_key] += regret

            # Store in advantage memory for Deep CFR training
            # Paper: Insert (I, t, r̃_t(I)) into MV,p
            if collect_deep_cfr_samples:
                self.advantage_memory[traversing_player].append({
                    'infoset': infoset,
                    'iteration': self.current_iteration,
                    'regrets': instantaneous_regrets,
                    'player': traversing_player
                })

            return node_value
        else:
            # Opponent's turn: sample action from strategy
            action_keys = [self.action_to_key(
                a, state, active_player) for a in legal_actions]
            probs = [strategy[key] for key in action_keys]
            sampled_action = random.choices(
                legal_actions, weights=probs, k=1)[0]

            # Store strategy sample for strategy network (MΠ)
            # Paper: Insert (I, t, σ_t(I)) into MΠ
            if collect_deep_cfr_samples:
                self.strategy_memory.append({
                    'infoset': infoset,
                    'iteration': self.current_iteration,
                    'strategy': dict(strategy),  # Copy the strategy
                    'player': active_player
                })

            # Recurse with sampled action
            next_state = state.proceed(sampled_action)
            value = self.external_sampling(next_state, traversing_player,
                                           collect_deep_cfr_samples, _depth + 1)

            # Update cumulative strategy (for legacy average strategy computation)
            for action in legal_actions:
                action_key = self.action_to_key(action, state, active_player)
                self.strategy_table[infoset][action_key] += strategy[action_key]

            return value

    def external_sampling_with_network(self, state, traversing_player: int,
                                       network_integrations: dict,
                                       collect_deep_cfr_samples: bool = True,
                                       _depth: int = 0) -> float:
        """
        External sampling MCCFR using neural networks for regret prediction.

        This is the proper Deep CFR algorithm per the paper:
        - At each infoset, use the NETWORK to predict regrets
        - Apply regret matching to get strategy
        - Explore according to that strategy

        Args:
            state: Current game state (RoundState or TerminalState)
            traversing_player: Player whose regrets we're updating (0 or 1)
            network_integrations: Dict mapping player -> NetworkMCCFRIntegration
                                  {0: integration_p0, 1: integration_p1}
            collect_deep_cfr_samples: If True, collect samples for training
            _depth: Current recursion depth (for preventing infinite loops)

        Returns:
            Utility value for traversing player at this node
        """
        # Terminal state: return utility (check early to avoid accessing non-existent attributes)
        if is_terminal_state(state):
            return float(state.deltas[traversing_player])

        # Depth limit: truncate extremely deep game trees
        # Return 0 (break-even estimate) when tree is too deep
        if _depth >= MCCFR.MAX_TRAVERSAL_DEPTH:
            return 0.0

        # Get information set
        active_player = state.button % 2
        state_id = id(state)

        if state_id in self.infoset_cache:
            infoset = self.infoset_cache[state_id]
        else:
            infoset = self.get_infoset(state, active_player)
            self.infoset_cache[state_id] = infoset

        # Get legal actions
        legal_actions = self.get_legal_actions_list(state)
        if not legal_actions:
            return 0.0

        # KEY DIFFERENCE: Use NETWORK to predict regrets, not cumulative table!
        # Paper Algorithm 2: "Compute strategy σt(I) from predicted advantages V(I(h), a|θp)"
        network_integration = network_integrations[active_player]
        # Pass precomputed legal_actions to avoid redundant expensive recomputation
        predicted_regrets = network_integration.get_network_regrets(
            state, active_player, legal_actions=legal_actions)

        # Get strategy via regret matching on network predictions
        strategy = self.regret_matching(
            predicted_regrets,  # Network predictions, not self.regret_table!
            legal_actions,
            state,
            active_player
        )

        # Check if it's the traversing player's turn
        if active_player == traversing_player:
            # Traversing player: compute regrets for ALL actions
            action_values = {}
            node_value = 0.0

            # Compute value of each action (explore all)
            for action in legal_actions:
                action_key = self.action_to_key(action, state, active_player)
                next_state = state.proceed(action)

                # CRITICAL FIX: Check if proceed() actually advanced the game
                if id(next_state) == id(state):
                    raise RuntimeError(
                        f"proceed() returned the same object (no progress)! "
                        f"Action: {action_key}, depth: {_depth}, street: {state.street}"
                    )

                # Check for progress signature (street, button, pips, stacks, board)
                # Skip check for terminal states (they don't have these attributes)
                if not is_terminal_state(next_state):
                    progress_sig = (
                        state.street, state.button % 2, tuple(state.pips),
                        tuple(state.stacks), len(state.board) if hasattr(
                            state, 'board') else 0
                    )
                    next_sig = (
                        next_state.street, next_state.button % 2, tuple(
                            next_state.pips),
                        tuple(next_state.stacks), len(next_state.board) if hasattr(
                            next_state, 'board') else 0
                    )

                    # Note: Identical signature might indicate a bug, but we don't raise
                    # an error here as it could be valid in some edge cases

                action_values[action_key] = self.external_sampling_with_network(
                    next_state, traversing_player, network_integrations, collect_deep_cfr_samples,
                    _depth + 1
                )
                node_value += strategy[action_key] * action_values[action_key]

            # Compute INSTANTANEOUS regrets
            # Paper: r̃_t(I,a) = v(a) - Σ σ(a')·v(a')
            instantaneous_regrets = {}
            for action in legal_actions:
                action_key = self.action_to_key(action, state, active_player)
                regret = action_values[action_key] - node_value
                instantaneous_regrets[action_key] = regret

            # Store in advantage memory for training
            if collect_deep_cfr_samples:
                self.advantage_memory[traversing_player].append({
                    'infoset': infoset,
                    'iteration': self.current_iteration,
                    'regrets': instantaneous_regrets,
                    'player': traversing_player
                })

            return node_value
        else:
            # Opponent's turn: sample ONE action from strategy
            action_keys = [self.action_to_key(
                a, state, active_player) for a in legal_actions]
            probs = [strategy[key] for key in action_keys]
            sampled_action = random.choices(
                legal_actions, weights=probs, k=1)[0]

            # Store strategy sample for strategy network (MΠ)
            if collect_deep_cfr_samples:
                self.strategy_memory.append({
                    'infoset': infoset,
                    'iteration': self.current_iteration,
                    'strategy': dict(strategy),
                    'player': active_player
                })

            # Recurse with sampled action
            next_state = state.proceed(sampled_action)

            # CRITICAL FIX: Check if proceed() actually advanced the game
            if id(next_state) == id(state):
                raise RuntimeError(
                    f"proceed() returned the same object (no progress)! "
                    f"Action: sampled, depth: {_depth}, street: {state.street}"
                )

            return self.external_sampling_with_network(
                next_state, traversing_player, network_integrations, collect_deep_cfr_samples,
                _depth + 1
            )

    def set_iteration(self, iteration: int):
        """Set the current CFR iteration number (for linear weighting)."""
        self.current_iteration = iteration

    def get_advantage_samples(self, player: int) -> list:
        """Get collected advantage samples for a player."""
        return self.advantage_memory[player]

    def get_strategy_samples(self) -> list:
        """Get collected strategy samples for the strategy network."""
        return self.strategy_memory

    def clear_advantage_memory(self, player: int = None):
        """Clear advantage memory (optionally for specific player)."""
        if player is not None:
            self.advantage_memory[player] = []
        else:
            self.advantage_memory = {0: [], 1: []}

    def clear_strategy_memory(self):
        """Clear strategy memory."""
        self.strategy_memory = []

    def clear_infoset_cache(self):
        """Clear infoset cache to free memory between iterations."""
        self.infoset_cache = {}

    def create_initial_state(self) -> RoundState:
        """
        Create a new game state with random cards dealt.

        Returns:
            RoundState at the start of preflop betting
        """
        deck = Deck()
        deck.shuffle()

        # Deal 3 cards to each player (MIT 2026 variant)
        hands = [deck.deal(3), deck.deal(3)]

        initial_state = RoundState(
            button=0,
            street=0,  # Preflop
            pips=[SMALL_BLIND, BIG_BLIND],
            stacks=[STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND],
            hands=hands,
            deck=deck,
            board=[],
            previous_state=None,
            action_taken=None
        )

        return initial_state

    def train(self, num_iterations: int, verbose: bool = True):
        """
        Train MCCFR by running multiple iterations.

        Each iteration:
        1. Create new game with random cards
        2. Run external sampling for Player 0
        3. Run external sampling for Player 1

        Args:
            num_iterations: Number of hands to play
            verbose: Whether to print progress
        """
        import time

        utilities = []
        iteration_times = []
        start_time = time.time()

        for iteration in range(num_iterations):
            iter_start = time.time()

            # Set current iteration for linear weighting
            self.set_iteration(iteration + 1)

            # Create new game
            initial_state = self.create_initial_state()

            # Traverse from both players' perspectives
            utility_p0 = self.external_sampling(
                initial_state, traversing_player=0)
            utility_p1 = self.external_sampling(
                initial_state, traversing_player=1)

            utilities.append((utility_p0, utility_p1))
            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)

            # Print progress
            if verbose and (iteration + 1) % 100 == 0:
                avg_p0 = np.mean([u[0] for u in utilities[-100:]])
                avg_p1 = np.mean([u[1] for u in utilities[-100:]])
                avg_iter_time = np.mean(iteration_times[-100:])
                total_elapsed = time.time() - start_time
                estimated_remaining = avg_iter_time * \
                    (num_iterations - iteration - 1)

                print(f"Iteration {iteration + 1}/{num_iterations}")
                print(f"  Avg utility P0: {avg_p0:.2f} chips/hand")
                print(f"  Avg utility P1: {avg_p1:.2f} chips/hand")
                print(f"  Infosets learned: {len(self.regret_table)}")
                print(
                    f"  Avg time/iter: {avg_iter_time:.3f}s ({1/avg_iter_time:.1f} iter/s)")
                print(
                    f"  Elapsed: {total_elapsed:.1f}s | ETA: {estimated_remaining:.1f}s")

        total_time = time.time() - start_time
        if verbose:
            print(f"\n✓ Training Complete!")
            print(f"  Total iterations: {num_iterations}")
            print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"  Avg time/iteration: {np.mean(iteration_times):.3f}s")
            print(f"  Information sets: {len(self.regret_table)}")
            print(f"  Strategy entries: {len(self.strategy_table)}")

        return utilities

    def get_average_strategy(self, infoset: str, legal_actions: List,
                             state: Optional[RoundState] = None,
                             active_player: Optional[int] = None) -> Dict[str, float]:
        """
        Get average strategy (Nash equilibrium approximation).

        The average strategy converges to Nash equilibrium, not the
        instantaneous regret-matching strategy.

        Args:
            infoset: Information set string
            legal_actions: List of legal action instances
            state: Current RoundState (for action encoding)
            active_player: Player index (for action encoding)

        Returns:
            Dictionary mapping action_key -> probability
        """
        strategy = {}
        action_keys = [self.action_to_key(
            a, state, active_player) for a in legal_actions]

        # Get cumulative strategy values
        cumulative_strategy = {key: self.strategy_table[infoset].get(key, 0.0)
                               for key in action_keys}

        # Normalize to get probabilities
        sum_strategy = sum(cumulative_strategy.values())
        if sum_strategy > 0:
            strategy = {
                key: cumulative_strategy[key] / sum_strategy for key in action_keys}
        else:
            # Uniform if no strategy accumulated
            uniform_prob = 1.0 / len(action_keys) if action_keys else 0.0
            strategy = {key: uniform_prob for key in action_keys}

        return strategy

    def select_action(self, state: RoundState, player: int,
                      use_average: bool = True):
        """
        Select an action for a player at a given state.

        Args:
            state: Current game state
            player: Player index (0 or 1)
            use_average: If True, use average strategy (Nash equilibrium)
                        If False, use current regret-matching strategy

        Returns:
            Action to take
        """
        infoset = self.get_infoset(state, player)
        legal_actions = self.get_legal_actions_list(state)

        if not legal_actions:
            return None

        if use_average:
            strategy = self.get_average_strategy(
                infoset, legal_actions, state, player)
        else:
            strategy = self.regret_matching(
                self.regret_table[infoset], legal_actions, state, player
            )

        # Sample action according to strategy
        action_keys = [self.action_to_key(
            a, state, player) for a in legal_actions]
        probs = [strategy[key] for key in action_keys]
        selected_action = random.choices(legal_actions, weights=probs, k=1)[0]

        return selected_action
