"""
Small masked policy/Q networks for discrete action spaces.
"""
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply softmax with mask (mask=0 -> -inf)."""
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(mask > 0, logits, torch.full_like(logits, neg_inf))
    return F.softmax(masked, dim=dim)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, ...] = (128, 128)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedPolicy(nn.Module):
    """
    Produces action logits; caller applies masked_softmax to sample or pick argmax.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: Tuple[int, ...] = (128, 128)):
        super().__init__()
        self.body = MLP(obs_dim, action_dim, hidden)

    def forward(self, obs: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        logits = self.body(obs)
        return masked_softmax(logits, legal_mask, dim=-1)


class MaskedQ(nn.Module):
    """
    Q-network with masked argmax for epsilon-greedy.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: Tuple[int, ...] = (128, 128)):
        super().__init__()
        self.body = MLP(obs_dim, action_dim, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.body(obs)

    def masked_argmax(self, obs: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        q = self.forward(obs)
        masked_q = q.masked_fill(legal_mask <= 0, torch.finfo(q.dtype).min)
        return masked_q.argmax(dim=-1)

