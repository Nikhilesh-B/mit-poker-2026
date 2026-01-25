"""
Self-play loop wiring the Env and NFSP agent together.
This is intentionally small so you can iterate quickly.
"""
from __future__ import annotations

import random
from typing import List

import torch

from .env import PokerEnv, ActionId, STARTING_STACK
from .nfsp import NFSPAgent, NFSPConfig
from .buffers import Transition


MAX_BOARD = 7  # flop(2 peek) + 2 discards + turn + river = 6, keep 7 for safety
MAX_HAND = 3   # before discards


def encode_obs(obs: dict) -> torch.Tensor:
    """
    Simple numeric encoding; pad variable-length card ids with -1.
    """
    board = obs["board_ids"][:MAX_BOARD] + [-1] * (MAX_BOARD - len(obs["board_ids"]))
    hand = obs["hand_ids"][:MAX_HAND] + [-1] * (MAX_HAND - len(obs["hand_ids"]))
    feats: List[float] = []
    feats.append(obs["street"] / 6.0)
    feats.append(obs["button"] % 2)
    feats.extend([p / STARTING_STACK for p in obs["pips"]])  # Fix: normalize by stack size
    feats.extend([s / STARTING_STACK for s in obs["stacks"]])
    feats.append(obs["pot"] / (2 * STARTING_STACK))
    feats.extend([(c + 1) / 52.0 for c in hand])
    feats.extend([(c + 1) / 52.0 for c in board])
    return torch.tensor(feats, dtype=torch.float32)


def legal_mask_vec(legal: List[ActionId]) -> torch.Tensor:
    mask = torch.zeros(len(ActionId), dtype=torch.float32)
    for a in legal:
        mask[int(a)] = 1.0
    return mask


def run_episode(env: PokerEnv, agent: NFSPAgent, eta_avg: float = 0.1, swap_seats: bool = False) -> float:
    """
    Runs one episode of self-play. Returns reward from player-0 perspective.
    
    Args:
        swap_seats: if True, flip button/positions to balance training across seats
    """
    obs = env.reset()
    done = False
    total_reward_p0 = 0.0
    while not done:
        active_seat = obs["button"] % 2
        active_is_p0 = (active_seat == 0) if not swap_seats else (active_seat == 1)
        o_t = encode_obs(obs)
        legal_mask = legal_mask_vec(obs["legal_actions"])
        use_avg = random.random() < eta_avg
        action = agent.select_action(o_t, legal_mask, use_avg=use_avg)

        next_obs, reward_p0, done, info = env.step(ActionId(action))
        # Flip reward if seats are swapped
        if swap_seats:
            reward_p0 = -reward_p0
        reward_active = reward_p0 if active_is_p0 else -reward_p0

        next_o_t = encode_obs(next_obs)
        next_mask = legal_mask_vec(next_obs["legal_actions"])
        transition = Transition(
            obs=o_t,
            legal_mask=legal_mask,
            action=action,
            reward=reward_active,
            done=done,
            next_obs=next_o_t,
            next_legal_mask=next_mask,
        )
        # RL buffer always gets the behavior transition (needed for Q targets)
        agent.push_transition(transition, to_sl=False)
        # NFSP: SL buffer should capture BR actions (imitation of best-response policy)
        if not use_avg:
            agent.push_transition(transition, to_sl=True)

        agent.train_step()
        obs = next_obs
        total_reward_p0 += reward_p0 if done else 0.0
    return total_reward_p0


def quick_smoke_test(episodes: int = 10):
    """
    Run a handful of episodes to verify plumbing.
    """
    env = PokerEnv()
    obs_dim = len(encode_obs(env.reset()))
    cfg = NFSPConfig(obs_dim=obs_dim, action_dim=len(ActionId))
    agent = NFSPAgent(cfg)
    bankroll = 0.0
    for ep in range(episodes):
        r = run_episode(env, agent, eta_avg=0.1)
        bankroll += r
        if (ep + 1) % 1 == 0:
            print(f"ep {ep+1} reward_p0 {r:+.1f} bankroll {bankroll:+.1f}")


if __name__ == "__main__":
    quick_smoke_test(episodes=5)

