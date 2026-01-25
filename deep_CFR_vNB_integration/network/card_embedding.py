import torch
import torch.nn as nn
import torch.nn.functional as F


class CardEmbedding(nn.Module):
    # dimensionality is for the cards we are talking about whether these are the hole cards or the board cards
    def __init__(self, dim):
        super(CardEmbedding, self).__init__()
        # Card encoding: rank * 10 + suit, max is 14*10 + 3 = 143, use 150 for safety
        self.card = nn.Embedding(150, dim)
        # Ranks 2-14 (Ace), so 15 possible values
        self.rank = nn.Embedding(15, dim)
        self.suit = nn.Embedding(4, dim)   # 4 suits (0-3)

    def forward(self, input):
        B, num_cards = input.shape
        flattened_cards = input.view(-1)

        # -1 means no card we can force the hand to be 3 and the board to be 6 and then with -1 in the missing spots
        valid_flattened_cards = flattened_cards.ge(0).float()
        flattened_cards = flattened_cards.clamp(min=0)

        # Extract rank and suit from encoding: rank = card // 10, suit = card % 10
        # But wait, looking at canon_cards.py, the encoding is rank*10 + suit where suit is 0-3
        # So rank = card // 10, suit = card % 10
        ranks = (flattened_cards // 10).clamp(max=14)  # Rank is 2-14
        suits = (flattened_cards % 10).clamp(max=3)     # Suit is 0-3

        embs = self.card(flattened_cards) + self.rank(ranks) + self.suit(suits)
        embs = embs * valid_flattened_cards.unsqueeze(1)

        return embs.view(B, num_cards, -1).sum(1)
