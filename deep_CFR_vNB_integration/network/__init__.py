"""
Neural Network Components for Deep CFR

This package contains:
- model: DeepCFRModule neural network architecture
- card_embedding: CardEmbedding layer for card representation
"""

from network.model import DeepCFRModule
from network.card_embedding import CardEmbedding

__all__ = [
    'DeepCFRModule',
    'CardEmbedding'
]
