"""
Minimal fast environment that mirrors engine.py rules for the discard variant.

Goals:
- no sockets; step/reset API for self-play.
- discrete raise grid to keep branching small for RL.
- uses pkrbot.Deck/evaluate for exact hand strength.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Dict, Any

import pkrbot

# Game constants (mirror config.py/engine.py)
STARTING_STACK = 400
BIG_BLIND = 2
SMALL_BLIND = 1


class ActionId(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE_SMALL = 2
    RAISE_MED = 3
    RAISE_LARGE = 4
    DISCARD_0 = 5
    DISCARD_1 = 6
    DISCARD_2 = 7


def card_index_map() -> Dict[str, int]:
    """Create a deterministic mapping from card string -> index [0,51]."""
    deck = pkrbot.Deck()
    return {str(card): idx for idx, card in enumerate(deck.cards)}


CARD_TO_ID = card_index_map()


@dataclass
class EnvState:
    button: int
    street: int
    pips: List[int]
    stacks: List[int]
    hands: List[List[pkrbot.Card]]
    deck: pkrbot.Deck
    board: List[pkrbot.Card]
    previous_state: "EnvState | None" = None

    # Helper to copy while updating some fields
    def copy(self, **kwargs) -> "EnvState":
        fields = dict(
            button=self.button,
            street=self.street,
            pips=list(self.pips),
            stacks=list(self.stacks),
            hands=[list(h) for h in self.hands],
            deck=self.deck,
            board=list(self.board),
            previous_state=self.previous_state,
        )
        fields.update(kwargs)
        return EnvState(**fields)


class PokerEnv:
    """
    Lightweight environment for self-play training.

    Streets:
      0: preflop betting (3-card hands)
      2: discard 1 (player 1 must discard to board; player 0 auto-checks)
      3: discard 2 (player 0 must discard; player 1 auto-checks)
      4: turn betting
      5: river betting
      6: showdown
    """

    def __init__(self, raise_grid: Tuple[float, float, float] = (1.0, 0.5, 1.0)):
        """
        Args:
            raise_grid: (small_factor, mid_factor, large_factor) applied to
                        (min_raise, max_raise) to pick discrete sizes.
                        mid is blended between min/max.
        """
        self.raise_grid = raise_grid
        self.state: EnvState | None = None
        self.done: bool = False
        self.last_action: ActionId | None = None

    # ---------- Core env API ----------
    def reset(self, rng: random.Random | None = None) -> Dict[str, Any]:
        rng = rng or random
        deck = pkrbot.Deck()
        deck.shuffle()
        hands = [deck.deal(3), deck.deal(3)]
        board: List[pkrbot.Card] = []
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        self.state = EnvState(
            button=0, street=0, pips=pips, stacks=stacks, hands=hands, deck=deck, board=board
        )
        self.done = False
        self.last_action = None
        return self._obs()

    def step(self, action_id: ActionId) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert self.state is not None, "Call reset() first"
        assert not self.done, "Episode already finished"
        state = self.state
        active = state.button % 2
        legal = self.legal_actions()
        if action_id not in legal:
            # Force a legal default to avoid crashing during exploration
            action_id = ActionId.CHECK_CALL if ActionId.CHECK_CALL in legal else list(legal)[0]

        # Map discrete action to concrete engine move
        reward = 0.0
        info: Dict[str, Any] = {}
        self.last_action = action_id

        if action_id in (ActionId.DISCARD_0, ActionId.DISCARD_1, ActionId.DISCARD_2):
            idx = int(action_id - ActionId.DISCARD_0)
            if len(state.hands[active]) > idx:
                card = state.hands[active].pop(idx)
                state.board.append(card)
            # advance to next player on same street (discard street auto-advances via proceed_street)
            next_state = state.copy(
                button=(1 - active) % 2,
                hands=state.hands,
                board=state.board,
                previous_state=state,
            )
            self.state = next_state
            # After both discards, proceed_street logic happens when the other player checks
            return self._advance_if_ready()

        # Fold
        if action_id == ActionId.FOLD:
            delta = self._get_delta(state, winner_index=(1 - active))
            self.done = True
            reward = delta if active == 0 else -delta
            self.state = EnvState(
                button=state.button,
                street=6,
                pips=state.pips,
                stacks=state.stacks,
                hands=state.hands,
                deck=state.deck,
                board=state.board,
                previous_state=state,
            )
            info["terminal_deltas"] = [delta, -delta]
            return self._obs(), reward, True, info

        # Call / Check
        if action_id == ActionId.CHECK_CALL:
            if state.button == 0 and state.street == 0:
                # SB calls BB -> both contributed BIG_BLIND
                next_state = EnvState(
                    button=1,
                    street=0,
                    pips=[BIG_BLIND, BIG_BLIND],
                    stacks=[STARTING_STACK - BIG_BLIND, STARTING_STACK - BIG_BLIND],
                    hands=state.hands,
                    deck=state.deck,
                    board=state.board,
                    previous_state=state,
                )
                self.state = next_state
                return self._advance_if_ready()
            # regular call/check
            new_pips = list(state.pips)
            new_stacks = list(state.stacks)
            contribution = new_pips[1 - active] - new_pips[active]
            new_stacks[active] -= contribution
            new_pips[active] += contribution
            next_state = EnvState(
                button=state.button + 1,
                street=state.street,
                pips=new_pips,
                stacks=new_stacks,
                hands=state.hands,
                deck=state.deck,
                board=state.board,
                previous_state=state,
            )
            self.state = next_state
            return self._advance_if_ready()

        # Raises
        if action_id in (ActionId.RAISE_SMALL, ActionId.RAISE_MED, ActionId.RAISE_LARGE):
            min_raise, max_raise = self._raise_bounds(state)
            amount = self._discrete_raise_amount(action_id, min_raise, max_raise)
            new_pips = list(state.pips)
            new_stacks = list(state.stacks)
            contribution = amount - new_pips[active]
            new_stacks[active] -= contribution
            new_pips[active] += contribution
            self.state = EnvState(
                button=state.button + 1,
                street=state.street,
                pips=new_pips,
                stacks=new_stacks,
                hands=state.hands,
                deck=state.deck,
                board=state.board,
                previous_state=state,
            )
            return self._obs(), 0.0, False, info

        raise ValueError(f"Unhandled action_id {action_id}")

    # ---------- Helpers ----------
    def legal_actions(self) -> List[ActionId]:
        """Return legal discrete actions for active player."""
        assert self.state is not None
        s = self.state
        active = s.button % 2
        continue_cost = s.pips[1 - active] - s.pips[active]

        if s.street in (2, 3):
            # Discard streets: only the designated player may discard; the other checks
            if active != s.street % 2:
                # must discard one of remaining hole cards
                return [
                    a
                    for a in (ActionId.DISCARD_0, ActionId.DISCARD_1, ActionId.DISCARD_2)
                    if (a - ActionId.DISCARD_0) < len(s.hands[active])
                ]
            return [ActionId.CHECK_CALL]

        actions: List[ActionId] = []
        if continue_cost == 0:
            actions.append(ActionId.CHECK_CALL)
            bets_forbidden = (s.stacks[0] == 0 or s.stacks[1] == 0)
            if not bets_forbidden:
                actions.extend([ActionId.RAISE_SMALL, ActionId.RAISE_MED, ActionId.RAISE_LARGE])
            actions.append(ActionId.FOLD)
            return actions

        # continue_cost > 0
        raises_forbidden = (continue_cost == s.stacks[active] or s.stacks[1 - active] == 0)
        actions.extend([ActionId.CHECK_CALL])  # treated as call here
        if not raises_forbidden:
            actions.extend([ActionId.RAISE_SMALL, ActionId.RAISE_MED, ActionId.RAISE_LARGE])
        actions.append(ActionId.FOLD)
        return actions

    def _advance_if_ready(self) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Advance streets when both players have acted, handle showdown."""
        assert self.state is not None
        s = self.state
        last_was_check = self.last_action == ActionId.CHECK_CALL
        discard_check_done = (s.street in (2, 3)) and last_was_check
        # Proceed when:
        # - both acted preflop (button>0 on street 0)
        # - both acted on betting streets (button>1)
        # - discard streets: only after the checker acts (last action was check/call)
        if (s.street == 0 and s.button > 0) or s.button > 1 or discard_check_done:
            next_state = self._proceed_street(s)
            self.state = next_state
        else:
            next_state = s

        if next_state.street == 6:
            delta = self._resolve_showdown(next_state)
            self.done = True
            reward = delta  # reward for player 0 perspective
            info = {"terminal_deltas": [delta, -delta]}
            return self._obs(), reward, True, info
        return self._obs(), 0.0, False, {}

    def _proceed_street(self, s: EnvState) -> EnvState:
        if s.street == 6:
            return s
        if s.street == 0:
            new_street = 2
            button = 1  # Player 1 discards first
            new_board = list(s.board)
            new_board.extend(s.deck.peek(new_street))  # two peeked cards
        elif s.street == 2:
            new_street = 3
            button = 0
            new_board = s.board
        elif s.street == 3:
            new_street = 4
            button = 1
            new_board = s.board
        else:
            new_street = s.street + 1
            button = 1
            new_board = list(s.board)
            new_board.append(s.deck.peek(new_street - 1)[new_street - 2])

        return EnvState(
            button=button,
            street=new_street,
            pips=[0, 0],
            stacks=s.stacks,
            hands=s.hands,
            deck=s.deck,
            board=new_board,
            previous_state=s,
        )

    def _resolve_showdown(self, s: EnvState) -> float:
        score0 = pkrbot.evaluate(s.board + s.hands[0])
        score1 = pkrbot.evaluate(s.board + s.hands[1])
        assert s.stacks[0] == s.stacks[1]
        if score0 > score1:
            delta = self._get_delta(s, 0)
        elif score1 > score0:
            delta = self._get_delta(s, 1)
        else:
            delta = self._get_delta(s, 2)
        return float(delta)

    def _get_delta(self, s: EnvState, winner_index: int) -> int:
        assert winner_index in (0, 1, 2)
        if winner_index == 2:
            delta = 0
        elif winner_index == 0:
            delta = STARTING_STACK - s.stacks[1]
        else:
            delta = s.stacks[0] - STARTING_STACK
        # Round toward player in position if fractional (match engine logic)
        if abs(delta - math.floor(delta)) > 1e-6:
            delta = math.floor(delta) if s.button % 2 == 0 else math.ceil(delta)
        return int(delta)

    def _raise_bounds(self, s: EnvState) -> Tuple[int, int]:
        active = s.button % 2
        continue_cost = s.pips[1 - active] - s.pips[active]
        max_contribution = min(s.stacks[active], s.stacks[1 - active] + continue_cost)
        min_contribution = min(max_contribution, continue_cost + max(continue_cost, BIG_BLIND))
        return (s.pips[active] + min_contribution, s.pips[active] + max_contribution)

    def _discrete_raise_amount(self, action_id: ActionId, min_raise: int, max_raise: int) -> int:
        if max_raise == min_raise:
            return max_raise
        if action_id == ActionId.RAISE_SMALL:
            return min_raise
        if action_id == ActionId.RAISE_LARGE:
            return max_raise
        # mid
        mid = int(round((min_raise + max_raise) / 2))
        return max(min_raise, min(max_raise, mid))

    # ---------- Observation ----------
    def _obs(self) -> Dict[str, Any]:
        assert self.state is not None
        s = self.state
        obs = {
            "street": s.street,
            "button": s.button,
            "pips": list(s.pips),
            "stacks": list(s.stacks),
            "pot": STARTING_STACK * 2 - s.stacks[0] - s.stacks[1],
            "board_ids": [CARD_TO_ID[str(c)] for c in s.board],
            "hand_ids": [CARD_TO_ID[str(c)] for c in s.hands[s.button % 2]],
            "legal_actions": self.legal_actions(),
            "last_action": self.last_action,
        }
        return obs

