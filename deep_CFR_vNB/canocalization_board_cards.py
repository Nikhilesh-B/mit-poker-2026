from typing import List, Tuple
import torch

# Rank ordering for sorting (Ace is highest)
RANK_ORDER = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
              '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


class canon_cards():
    def __init__(self, hand, board):
        self.hand = hand
        self.board = board
        self.suit_mapping = {}
        self.next_canonical_suit = 0
        self.canonical_hand, self.canonical_board = self.canonicalize_cards(
            hand, board)

    def get_card_suit(self, card) -> str:
        card_str = str(card)
        return card_str[-1]

    def get_card_rank(self, card) -> str:
        card_str = str(card)
        return card_str[:-1]

    def canonicalize_card(self, card) -> str:
        rank = self.get_card_rank(card)
        suit = self.get_card_suit(card)
        canonical_suit = self.suit_mapping[suit]
        return f"{rank}s{canonical_suit}"

    def canonicalize_cards(self, hand: List, board: List) -> Tuple[List[str], List[str]]:
        all_cards = []
        for i, card in enumerate(hand):
            all_cards.append(('hand', i, card))
        for i, card in enumerate(board):
            all_cards.append(('board', i, card))

        # Sort by rank (descending: Ace=14 down to 2=2)
        all_cards.sort(
            key=lambda x: RANK_ORDER[self.get_card_rank(x[2])], reverse=True)

        # Build suit mapping by processing in rank order

        for source, idx, card in all_cards:
            suit = self.get_card_suit(card)
            if suit not in self.suit_mapping:
                self.suit_mapping[suit] = self.next_canonical_suit
                self.next_canonical_suit += 1

        canonical_hand = [self.canonicalize_card(c) for c in hand]
        canonical_board = [self.canonicalize_card(c) for c in board]

        return canonical_hand, canonical_board

    def add_card_board(self, card):
        canon_card = self.canonicalize_card(card)
        self.canonical_board.append(canon_card)

    def remove_card_hand(self, idx):
        removed_card = self.hand.pop(idx)
        self.canonical_board.append(removed_card)

    def get_board_str(self) -> str:
        board_str = ','.join(sorted(self.canonical_board))
        return board_str

    def get_hand_str(self) -> str:
        hand_str = ','.join(sorted(self.canonical_hand))
        return hand_str

    def encode_card_int(self, card) -> int:
        card_str = str(card)
        rank, suit = card_str.split('s')
        rank, suit = int(rank), int(suit)
        return rank*10+suit

    def get_canonical_hand_tensor(self) -> torch.tensor:
        canonical_hand_tensor = [
            self.encode_card_int(c) for c in sorted(self.hand)]
        return torch.tensor(canonical_hand_tensor)

    def get_canonical_board_tensor(self) -> torch.tensor:
        canonical_board_tensor = [
            self.encode_card_int(c) for c in sorted(self.board)]
        return torch.tensor(canonical_board_tensor)
