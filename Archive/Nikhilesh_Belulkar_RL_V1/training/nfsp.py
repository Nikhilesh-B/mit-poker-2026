"""
Minimal NFSP-style agent scaffolding for the discard variant.

This is intentionally lightweight:
- shared parameters for both seats
- epsilon-greedy Q for best-response policy
- supervised buffer for average policy cloning
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import random

import torch
import torch.nn as nn
import torch.optim as optim

from .policy import MaskedPolicy, MaskedQ, masked_softmax
from .buffers import ReplayBuffer, Transition


@dataclass
class NFSPConfig:
    obs_dim: int
    action_dim: int
    epsilon: float = 0.1
    gamma: float = 0.99
    rl_lr: float = 1e-3
    sl_lr: float = 1e-3
    batch_size: int = 64
    target_sync: int = 200
    rl_hidden: Tuple[int, ...] = (128, 128)
    sl_hidden: Tuple[int, ...] = (128, 128)
    rl_capacity: int = 50_000
    sl_capacity: int = 50_000


class NFSPAgent:
    """
    Very small NFSP agent: DQN for RL policy, policy net for average strategy.
    """

    def __init__(self, cfg: NFSPConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

        self.q = MaskedQ(cfg.obs_dim, cfg.action_dim, cfg.rl_hidden).to(self.device)
        self.q_target = MaskedQ(cfg.obs_dim, cfg.action_dim, cfg.rl_hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.pi = MaskedPolicy(cfg.obs_dim, cfg.action_dim, cfg.sl_hidden).to(self.device)

        self.q_opt = optim.Adam(self.q.parameters(), lr=cfg.rl_lr)
        self.pi_opt = optim.Adam(self.pi.parameters(), lr=cfg.sl_lr)

        self.rl_buffer = ReplayBuffer(cfg.rl_capacity)
        self.sl_buffer = ReplayBuffer(cfg.sl_capacity)
        self._step_count = 0

    def select_action(self, obs: torch.Tensor, legal_mask: torch.Tensor, use_avg: bool = False) -> int:
        """
        obs: shape [obs_dim]
        legal_mask: shape [action_dim] with 1 for legal, 0 otherwise
        """
        obs = obs.to(self.device)
        legal_mask = legal_mask.to(self.device)
        if use_avg:
            probs = self.pi(obs.unsqueeze(0), legal_mask.unsqueeze(0)).squeeze(0)
            dist = torch.distributions.Categorical(probs=probs)
            return int(dist.sample().item())

        if random.random() < self.cfg.epsilon:
            legal_idxs = torch.nonzero(legal_mask, as_tuple=False).squeeze(-1)
            return int(random.choice(legal_idxs.cpu().tolist()))

        with torch.no_grad():
            qvals = self.q(obs.unsqueeze(0))
            masked_q = qvals.masked_fill(legal_mask.unsqueeze(0) <= 0, torch.finfo(qvals.dtype).min)
            return int(masked_q.argmax(dim=-1).item())

    def push_transition(self, transition: Transition, to_sl: bool = False) -> None:
        if to_sl:
            self.sl_buffer.push(transition)
        else:
            self.rl_buffer.push(transition)

    def train_step(self) -> None:
        self._step_count += 1
        self._train_rl()
        self._train_sl()
        if self._step_count % self.cfg.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    def _train_rl(self) -> None:
        if len(self.rl_buffer) < self.cfg.batch_size:
            return
        batch = self.rl_buffer.sample(self.cfg.batch_size)
        obs = torch.stack([t.obs for t in batch]).to(self.device)
        mask = torch.stack([t.legal_mask for t in batch]).to(self.device)
        actions = torch.tensor([t.action for t in batch], device=self.device, dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], device=self.device, dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], device=self.device, dtype=torch.float32)
        next_obs = torch.stack([t.next_obs for t in batch]).to(self.device)
        next_mask = torch.stack([t.next_legal_mask for t in batch]).to(self.device)

        q_values = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # CRITICAL: Use target network for stable TD targets
            next_q = self.q_target(next_obs)
            next_q = next_q.masked_fill(next_mask <= 0, torch.finfo(next_q.dtype).min)
            next_best = next_q.max(dim=1).values
            target = rewards + (1.0 - dones) * self.cfg.gamma * next_best

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()

    def _train_sl(self) -> None:
        if len(self.sl_buffer) < self.cfg.batch_size:
            return
        batch = self.sl_buffer.sample(self.cfg.batch_size)
        obs = torch.stack([t.obs for t in batch]).to(self.device)
        mask = torch.stack([t.legal_mask for t in batch]).to(self.device)
        actions = torch.tensor([t.action for t in batch], device=self.device, dtype=torch.long)

        probs = self.pi(obs, mask)
        logp = torch.log(torch.clamp(probs.gather(1, actions.unsqueeze(1)).squeeze(1), min=1e-8))
        loss = -logp.mean()
        self.pi_opt.zero_grad()
        loss.backward()
        self.pi_opt.step()

