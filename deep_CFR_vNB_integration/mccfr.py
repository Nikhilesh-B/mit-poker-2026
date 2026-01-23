"""
MCCFR (Monte Carlo Counterfactual Regret Minimization) Implementation

This is a clean, object-oriented implementation of MCCFR that can be easily
integrated with Deep CFR networks.

Usage:
    mccfr = MCCFR()
    mccfr.train(num_iterations=1000)
    action = mccfr.select_action(state, player=0)
"""

import sys
import os

# Add parent directory to path for imports BEFORE importing custom_engine
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Now we can import from custom_engine
from custom_engine import (
    RoundState, TerminalState,
    FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction,
    STARTING_STACK, BIG_BLIND, SMALL_BLIND
)
from pkrbot import Deck
from canon_cards import canon_cards

import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional


class MCCFR:
    """
    Monte Carlo Counterfactual Regret Minimization (MCCFR) Algorithm

    This class implements Algorithm 4: External Sampling with 
    Stochastically-Weighted Averaging.

    The algorithm learns an approximate Nash equilibrium strategy by:
    1. Recursively traversing the game tree
    2. Computing counterfactual regrets at each decision node
    3. Updating regret tables (can be replaced with neural network)
    4. Converting regrets to strategies via regret matching
    """

    def __init__(self):
        """Initialize MCCFR with empty regret and strategy tables."""
        # Regret table: r_I[a] = cumulative regret for action a at infoset I
        self.regret_table = defaultdict(lambda: defaultdict(float))

        # Strategy table: s_I[a] = cumulative strategy for action a at infoset I
        self.strategy_table = defaultdict(lambda: defaultdict(float))

        # Cache for infoset lookups (avoids recomputation)
        self.infoset_cache = {}

    def get_infoset(self, state: RoundState, player: int) -> str:
        """
        Get information set string for a state.

        Information set = what the player knows at a decision point:
        - Street (game phase)
        - Their hand (canonical)
        - Board cards (canonical)
        - Action history

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

        # Extract action history from action_taken field
        history = []
        current = state
        while current.previous_state is not None:
            prev = current.previous_state

            if hasattr(current, 'action_taken') and current.action_taken is not None:
                action = current.action_taken

                if isinstance(action, FoldAction):
                    history.append('F')
                elif isinstance(action, CallAction):
                    history.append('C')
                elif isinstance(action, CheckAction):
                    history.append('X')
                elif isinstance(action, DiscardAction):
                    history.append('D')
                elif isinstance(action, RaiseAction):
                    # Bucket bet sizes into 3 categories
                    pot_from_previous_streets = (
                        2 * STARTING_STACK) - sum(prev.stacks)
                    pot_this_street = sum(prev.pips)
                    pot_before_bet = pot_from_previous_streets + pot_this_street

                    bet_amount = max(current.pips) - max(prev.pips)

                    if pot_before_bet > 0:
                        bet_to_pot_ratio = bet_amount / pot_before_bet
                        if bet_to_pot_ratio < 0.5:
                            history.append('r')  # Small bet
                        elif bet_to_pot_ratio < 1.0:
                            history.append('R')  # Medium bet
                        else:
                            history.append('B')  # Large bet
                    else:
                        history.append('R')  # Default to medium

            current = prev

        # Reverse to get chronological order
        history.reverse()
        history_str = ''.join(history[-20:])  # Keep last 20 actions

        # Combine into canonical infoset string
        infoset = f"S{street}|H:{hand_str}|B:{board_str}|A:{history_str}"
        return infoset

    def action_to_key(self, action, state: Optional[RoundState] = None,
                      active_player: Optional[int] = None) -> str:
        """
        Convert an action object to a string key.

        Args:
            action: Action instance (FoldAction, CallAction, etc.)
            state: Current RoundState (required for DiscardAction)
            active_player: Player index (required for DiscardAction)

        Returns:
            String key representing the action
        """
        if isinstance(action, FoldAction):
            return "FOLD"
        elif isinstance(action, CallAction):
            return "CALL"
        elif isinstance(action, CheckAction):
            return "CHECK"
        elif isinstance(action, RaiseAction):
            # Bucket raise by size relative to pot
            if state is not None:
                pot_from_previous_streets = (
                    2 * STARTING_STACK) - sum(state.stacks)
                pot_this_street = sum(state.pips)
                pot_before_bet = pot_from_previous_streets + pot_this_street

                current_pip = state.pips[active_player] if active_player is not None else 0
                bet_amount = action.amount - current_pip

                if pot_before_bet > 0:
                    bet_to_pot_ratio = bet_amount / pot_before_bet
                    if bet_to_pot_ratio < 0.5:
                        return "RAISE_SMALL"
                    elif bet_to_pot_ratio < 1.0:
                        return "RAISE_MEDIUM"
                    else:
                        return "RAISE_LARGE"
                else:
                    return "RAISE_MEDIUM"
            else:
                return f"RAISE_{action.amount}"
        elif isinstance(action, DiscardAction):
            # Encode by canonical card value, not position
            if state is not None and active_player is not None:
                cards = canon_cards(state.hands[active_player], state.board)
                original_card = state.hands[active_player][action.card]
                canonical_card = cards.canonicalize_card(original_card)
                return f"DISCARD_{canonical_card}"
            else:
                return f"DISCARD_{action.card}"
        elif isinstance(action, type):
            # Handle action types (not instances)
            if action == FoldAction:
                return "FOLD"
            elif action == CallAction:
                return "CALL"
            elif action == CheckAction:
                return "CHECK"
            elif action == RaiseAction:
                return "RAISE"
            elif action == DiscardAction:
                return "DISCARD"
        return str(action)

    def get_legal_actions_list(self, state: RoundState) -> List:
        """
        Get list of legal actions with concrete values.

        Action abstraction:
        - Raises: 3 discrete sizes (MIN, MID, MAX)
        - Discards: All cards in hand
        - Check/Call/Fold: Single action each

        Args:
            state: Current RoundState

        Returns:
            List of concrete action instances
        """
        legal_action_types = state.legal_actions()
        actions = []
        active = state.button % 2

        for action_type in legal_action_types:
            if action_type == RaiseAction:
                # Discretize raise space into 3 sizes
                min_raise, max_raise = state.raise_bounds()
                raise_sizes = [
                    min_raise,
                    (min_raise + max_raise) // 2,
                    max_raise
                ]
                raise_sizes = sorted(list(set(raise_sizes))
                                     )  # Remove duplicates

                for size in raise_sizes:
                    actions.append(RaiseAction(size))
            elif action_type == DiscardAction:
                # All cards in hand can be discarded
                for card_idx in range(len(state.hands[active])):
                    actions.append(DiscardAction(card_idx))
            else:
                # Fold, Call, Check
                actions.append(action_type())

        return actions

    def regret_matching(self, regrets: Dict[str, float], actions: List,
                        state: Optional[RoundState] = None,
                        active_player: Optional[int] = None) -> Dict[str, float]:
        """
        Convert regrets to strategy using regret matching.

        Algorithm:
        1. Take max(regret, 0) for each action (positive regrets only)
        2. Sum all positive regrets
        3. If sum > 0: normalize to get probabilities
        4. If sum = 0: uniform distribution over all actions

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

        # Step 3 & 4: Normalize or use uniform
        if sum_positive > 0:
            strategy = {
                key: positive_regrets[key] / sum_positive for key in action_keys}
        else:
            uniform_prob = 1.0 / len(action_keys) if action_keys else 0.0
            strategy = {key: uniform_prob for key in action_keys}

        return strategy

    def external_sampling(self, state: Union[RoundState, TerminalState],
                          traversing_player: int) -> float:
        """
        External sampling MCCFR traversal (core algorithm).

        This recursively traverses the game tree:
        - Terminal states: return utility
        - Traversing player's nodes: compute regrets, update regret table
        - Opponent's nodes: sample action, update strategy table

        Args:
            state: Current game state
            traversing_player: Player whose regrets we're updating (0 or 1)

        Returns:
            Utility value for traversing player at this node
        """
        # Terminal state: return utility
        if isinstance(state, TerminalState):
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
                    next_state, traversing_player
                )
                node_value += strategy[action_key] * action_values[action_key]

            # Compute regrets and update regret table
            for action in legal_actions:
                action_key = self.action_to_key(action, state, active_player)
                regret = action_values[action_key] - node_value
                self.regret_table[infoset][action_key] += regret

            return node_value
        else:
            # Opponent's turn: sample action from strategy
            action_keys = [self.action_to_key(
                a, state, active_player) for a in legal_actions]
            probs = [strategy[key] for key in action_keys]
            sampled_action = random.choices(
                legal_actions, weights=probs, k=1)[0]

            # Recurse with sampled action
            next_state = state.proceed(sampled_action)
            value = self.external_sampling(next_state, traversing_player)

            # Update cumulative strategy
            for action in legal_actions:
                action_key = self.action_to_key(action, state, active_player)
                self.strategy_table[infoset][action_key] += strategy[action_key]

            return value

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
