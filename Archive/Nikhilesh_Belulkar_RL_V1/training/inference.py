"""
Inference utilities to run a trained NFSP policy inside player.py.
"""
from __future__ import annotations

from typing import List, Tuple, Optional

import torch

from .env import ActionId, CARD_TO_ID, STARTING_STACK, BIG_BLIND
from .policy import MaskedPolicy
from .nfsp import NFSPConfig


def _discrete_raise_amount(action_id: ActionId, min_raise: int, max_raise: int) -> int:
    """Mirror env.py discrete raise selection."""
    if max_raise == min_raise:
        return max_raise
    if action_id == ActionId.RAISE_SMALL:
        return min_raise
    if action_id == ActionId.RAISE_LARGE:
        return max_raise
    # mid
    mid = int(round((min_raise + max_raise) / 2))
    return max(min_raise, min(max_raise, mid))


def build_mask(round_state, active: int) -> torch.Tensor:
    """Construct legal mask aligned to ActionId order, mirroring round_state.legal_actions()."""
    mask = torch.zeros(len(ActionId), dtype=torch.float32)
    legal = round_state.legal_actions()
    legal_names = {a.__name__ for a in legal}

    # Discard streets: only discard or check
    if "DiscardAction" in legal_names:
        for idx in range(len(round_state.hands[active])):
            mask[int(ActionId.DISCARD_0) + idx] = 1.0
        return mask
    # On discard streets, non-discard player can only check
    if round_state.street in (2, 3) and "CheckAction" in legal_names:
        mask[int(ActionId.CHECK_CALL)] = 1.0
        return mask

    continue_cost = round_state.pips[1 - active] - round_state.pips[active]
    min_raise, max_raise = round_state.raise_bounds() if hasattr(round_state, "raise_bounds") else (0, 0)

    # check/call branch
    if continue_cost == 0:
        mask[int(ActionId.CHECK_CALL)] = 1.0
        bets_forbidden = (round_state.stacks[0] == 0 or round_state.stacks[1] == 0)
        if not bets_forbidden and "RaiseAction" in legal_names:
            mask[int(ActionId.RAISE_SMALL)] = 1.0
            mask[int(ActionId.RAISE_MED)] = 1.0
            mask[int(ActionId.RAISE_LARGE)] = 1.0
        if "FoldAction" in legal_names:
            mask[int(ActionId.FOLD)] = 1.0
        return mask

    # facing bet
    if "CallAction" in legal_names:
        mask[int(ActionId.CHECK_CALL)] = 1.0  # call
    raises_forbidden = (continue_cost == round_state.stacks[active] or round_state.stacks[1 - active] == 0)
    if not raises_forbidden and "RaiseAction" in legal_names and min_raise <= max_raise:
        mask[int(ActionId.RAISE_SMALL)] = 1.0
        mask[int(ActionId.RAISE_MED)] = 1.0
        mask[int(ActionId.RAISE_LARGE)] = 1.0
    if "FoldAction" in legal_names:
        mask[int(ActionId.FOLD)] = 1.0
    return mask


def encode_obs(round_state, active: int, last_action: Optional[int]) -> torch.Tensor:
    """
    Match training encode_obs: uses card ids padded with -1, scaled scalars.
    """
    MAX_BOARD = 7
    MAX_HAND = 3
    board_cards = round_state.board if hasattr(round_state, "board") else []
    my_hand = round_state.hands[active]
    board_ids = [CARD_TO_ID.get(str(c), -1) for c in board_cards]
    hand_ids = [CARD_TO_ID.get(str(c), -1) for c in my_hand]
    board = board_ids[:MAX_BOARD] + [-1] * (MAX_BOARD - len(board_ids))
    hand = hand_ids[:MAX_HAND] + [-1] * (MAX_HAND - len(hand_ids))

    pot = STARTING_STACK * 2 - round_state.stacks[0] - round_state.stacks[1]
    feats: List[float] = []
    feats.append(round_state.street / 6.0)
    feats.append(round_state.button % 2)
    feats.extend([p / STARTING_STACK for p in round_state.pips])  # Fix: normalize by stack size
    feats.extend([s / STARTING_STACK for s in round_state.stacks])
    feats.append(pot / (2 * STARTING_STACK))
    feats.extend([(c + 1) / 52.0 for c in hand])
    feats.extend([(c + 1) / 52.0 for c in board])
    # last_action as a single scalar (optional); keep backward-compatible length by omitting
    return torch.tensor(feats, dtype=torch.float32)


def map_action_to_engine(action_id: int, round_state) -> object:
    """Translate ActionId to engine action object."""
    from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction  # local import

    aid = ActionId(action_id)
    active = round_state.button % 2
    continue_cost = round_state.pips[1 - active] - round_state.pips[active]

    if aid in (ActionId.DISCARD_0, ActionId.DISCARD_1, ActionId.DISCARD_2):
        idx = int(aid - ActionId.DISCARD_0)
        return DiscardAction(idx)
    if aid == ActionId.FOLD:
        return FoldAction()
    if aid == ActionId.CHECK_CALL:
        return CheckAction() if continue_cost == 0 else CallAction()
    if aid in (ActionId.RAISE_SMALL, ActionId.RAISE_MED, ActionId.RAISE_LARGE):
        min_raise, max_raise = round_state.raise_bounds()
        amount = _discrete_raise_amount(aid, min_raise, max_raise)
        return RaiseAction(amount)
    # Fallback safe check
    return CheckAction()


def load_policy(checkpoint_path: str, device: str = "cpu") -> Tuple[MaskedPolicy, NFSPConfig]:
    """Load policy network from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt["cfg"]
    cfg = NFSPConfig(**cfg_dict)
    pi = MaskedPolicy(cfg.obs_dim, cfg.action_dim, cfg.sl_hidden).to(device)
    pi.load_state_dict(ckpt["pi"])
    pi.eval()
    return pi, cfg

