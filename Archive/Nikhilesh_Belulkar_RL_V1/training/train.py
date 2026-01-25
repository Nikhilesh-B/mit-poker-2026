"""
Simple training script to run NFSP-style self-play on the discard variant.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from .env import PokerEnv, ActionId
from .nfsp import NFSPAgent, NFSPConfig
from .selfplay import run_episode, encode_obs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NFSP on discard-variant poker")
    parser.add_argument("--episodes", type=int, default=10_000, help="Number of training episodes (hands)")
    parser.add_argument("--eta_avg", type=float, default=0.1, help="Prob. of using average policy (NFSP eta)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-greedy best response")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--save_interval", type=int, default=2_000, help="Save checkpoint every N episodes")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(agent: NFSPAgent, path: Path, episode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "q": agent.q.state_dict(),
            "pi": agent.pi.state_dict(),
            "cfg": agent.cfg.__dict__,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    env = PokerEnv()
    obs_dim = len(encode_obs(env.reset()))
    cfg = NFSPConfig(
        obs_dim=obs_dim,
        action_dim=len(ActionId),
        epsilon=args.epsilon,
        rl_hidden=(128, 128),
        sl_hidden=(128, 128),
        batch_size=128,
        rl_capacity=100_000,
        sl_capacity=100_000,
        target_sync=500,
    )
    agent = NFSPAgent(cfg, device=args.device)

    save_dir = Path(args.save_dir)
    start_time = time.time()
    bankroll = 0.0
    for ep in range(1, args.episodes + 1):
        # Alternate starting positions to balance positional training
        swap_seats = (ep % 2 == 0)
        reward_p0 = run_episode(env, agent, eta_avg=args.eta_avg, swap_seats=swap_seats)
        bankroll += reward_p0
        if ep % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[ep {ep}] reward_p0={reward_p0:+.1f} bankroll={bankroll:+.1f} elapsed={elapsed:.1f}s")
        if args.save_interval > 0 and ep % args.save_interval == 0:
            ckpt_path = save_dir / f"nfsp_ep{ep}.pt"
            save_checkpoint(agent, ckpt_path, ep)
            print(f"Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()

