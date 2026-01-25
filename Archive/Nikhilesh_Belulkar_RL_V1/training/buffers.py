"""
Simple replay buffers for NFSP-style training.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple
import random
import torch


@dataclass
class Transition:
    obs: torch.Tensor
    legal_mask: torch.Tensor
    action: int
    reward: float
    done: bool
    next_obs: torch.Tensor
    next_legal_mask: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

