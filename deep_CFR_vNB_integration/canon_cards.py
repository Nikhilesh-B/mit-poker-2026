from typing import List, Tuple
import torch

# Rank ordering for sorting (Ace is highest)
RANK_ORDER = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
              '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


class canon_cards():
    def __init__(self, hand, board):
        """
        Initialize canonical card representation.

        Args:
            hand: List of Card objects (player's hole cards)
            board: List of Card objects (community cards)
        """
        self.suit_mapping = {}
        self.next_canonical_suit = 0
        self.canonical_hand, self.canonical_board = self.canonicalize_cards(
            hand, board)
        # Keep canonical arrays sorted at all times for consistency
        # Sort by numeric rank (descending: Ace=14 highest)
        self.canonical_hand = sorted(
            self.canonical_hand, key=lambda x: int(x.split('s')[0]), reverse=True)
        self.canonical_board = sorted(
            self.canonical_board, key=lambda x: int(x.split('s')[0]), reverse=True)

    def get_card_suit(self, card) -> str:
        card_str = str(card)
        return card_str[-1]

    def get_card_rank(self, card) -> str:
        card_str = str(card)
        return card_str[:-1]

    def canonicalize_card(self, card) -> str:
        rank = str(RANK_ORDER[self.get_card_rank(card)])
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
        """
        Add a card to the board (canonicalized) and maintain sorted order.

        NOTE: This method uses the existing suit mapping. For best results,
        canonicalize all cards together when initializing the object.

        Args:
            card: Card object to add to board
        """
        canon_card = self.canonicalize_card(card)
        self.canonical_board.append(canon_card)
        # Keep sorted for consistency (by numeric rank, descending)
        self.canonical_board = sorted(
            self.canonical_board, key=lambda x: int(x.split('s')[0]), reverse=True)

    def remove_card_hand_by_canonical(self, canonical_card: str):
        """
        Remove a canonical card from hand and add it to the board.
        Maintains sorted order of both arrays.

        NOTE: This method uses the existing suit mapping. For best results,
        create a new canon_cards object when cards change significantly.

        Args:
            canonical_card: Canonical card string to remove (e.g., "14s0")
        """
        if canonical_card in self.canonical_hand:
            self.canonical_hand.remove(canonical_card)
            self.canonical_board.append(canonical_card)
            # Keep sorted for consistency (by numeric rank, descending)
            self.canonical_board = sorted(
                self.canonical_board, key=lambda x: int(x.split('s')[0]), reverse=True)
            self.canonical_hand = sorted(
                self.canonical_hand, key=lambda x: int(x.split('s')[0]), reverse=True)

    def get_board_str(self) -> str:
        """
        Get sorted board string representation.
        Note: canonical_board is kept sorted internally, so no sorting needed.
        """
        return ','.join(self.canonical_board)

    def get_hand_str(self) -> str:
        """
        Get sorted hand string representation.
        Note: canonical_hand is kept sorted internally, so no sorting needed.
        """
        return ','.join(self.canonical_hand)

    def encode_card_int(self, card) -> int:
        """
        Encode a canonical card string (e.g., "14s0" for Ace with canonical suit 0) 
        into an integer representation.

        Format: "{rank}s{suit}" where rank is 2-14 (numeric) and suit is 0-3 (canonical suit)
        Encoding: rank * 10 + suit (e.g., "14s0" → 140, "2s3" → 23)

        Args:
            card: Canonical card string (e.g., "14s0", "13s1", "2s3")

        Returns:
            Integer encoding of the card
        """
        card_str = str(card)
        rank, suit = card_str.split('s')
        rank, suit = int(rank), int(suit)
        return rank * 10 + suit

    def get_canonical_hand_tensor(self) -> torch.tensor:
        """
        Get tensor representation of canonical hand.
        Note: canonical_hand is kept sorted internally, so no sorting needed.
        """
        canonical_hand_tensor = [
            self.encode_card_int(c) for c in self.canonical_hand]
        return torch.tensor(canonical_hand_tensor)

    def get_canonical_board_tensor(self) -> torch.tensor:
        """
        Get tensor representation of canonical board.
        Note: canonical_board is kept sorted internally, so no sorting needed.
        """
        canonical_board_tensor = [
            self.encode_card_int(c) for c in self.canonical_board]
        return torch.tensor(canonical_board_tensor)
