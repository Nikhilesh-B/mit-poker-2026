"""
Infoset Parser for Deep CFR Integration

This module parses MCCFR infoset strings and extracts features needed for the network:
- Canonical cards (hand + board)
- Action history string

Infoset format: "S{street}|H:{hand_str}|B:{board_str}|A:{history_str}"
Example: "S0|H:14s0,13s0,10s1|B:|A:RMC"
"""

import torch
from typing import Tuple, Dict
from utils.canon_cards import canon_cards


class CanonicalCardsFromString(canon_cards):
    """
    Extension of canon_cards that can be initialized directly from canonical strings.

    This is used when we parse an infoset string and need to recreate a canon_cards
    object without having the original Card objects.
    """

    def __init__(self, canonical_hand: list, canonical_board: list):
        """
        Initialize from canonical card strings directly.

        Args:
            canonical_hand: List of canonical card strings (e.g., ["14s0", "13s0"])
            canonical_board: List of canonical card strings (e.g., ["10s1", "9s0"])
        """
        # Skip parent __init__ since we don't have original Card objects
        # Instead, directly set the canonical representations
        self.canonical_hand = sorted(
            canonical_hand, key=lambda x: int(x.split('s')[0]), reverse=True)
        self.canonical_board = sorted(
            canonical_board, key=lambda x: int(x.split('s')[0]), reverse=True)

        # Reconstruct suit mapping from canonical cards
        self.suit_mapping = {}
        self.next_canonical_suit = 0

        # Extract all canonical suits used
        all_cards = canonical_hand + canonical_board
        canonical_suits = set()
        for card_str in all_cards:
            if 's' in card_str:
                suit = card_str.split('s')[1]
                canonical_suits.add(suit)

        # Create dummy suit mapping (we don't need the original suits)
        for canonical_suit in sorted(canonical_suits):
            self.suit_mapping[f'dummy_{canonical_suit}'] = canonical_suit
            self.next_canonical_suit = max(
                self.next_canonical_suit, int(canonical_suit) + 1)


def parse_infoset_string(infoset_str: str) -> Tuple[CanonicalCardsFromString, str, int]:
    """
    Parse MCCFR infoset string into components needed for network.

    Infoset format: "S{street}|H:{hand_str}|B:{board_str}|A:{history_str}"
    - Street: game phase (0-3)
    - Hand: canonical cards like "14s0,13s0,10s1"
    - Board: canonical cards like "10s1,9s0"
    - Action history: string like "RMC" (Raise Medium, Call, etc.)

    Args:
        infoset_str: Infoset string from MCCFR

    Returns:
        Tuple of (canon_cards_object, action_history_string, street)
    """
    # Split by main delimiters
    parts = infoset_str.split('|')

    # Extract street
    street_part = parts[0]  # "S0", "S1", etc.
    street = int(street_part[1:])

    # Extract hand string
    hand_part = parts[1]  # "H:14s0,13s0,10s1"
    hand_str = hand_part[2:]  # Remove "H:"
    canonical_hand = hand_str.split(',') if hand_str else []

    # Extract board string
    board_part = parts[2]  # "B:10s1,9s0" or "B:"
    board_str = board_part[2:]  # Remove "B:"
    canonical_board = board_str.split(',') if board_str else []

    # Extract action history
    action_part = parts[3]  # "A:RMC"
    action_history = action_part[2:]  # Remove "A:"

    # Create canon_cards object from canonical strings
    canon_cards_obj = CanonicalCardsFromString(canonical_hand, canonical_board)

    return canon_cards_obj, action_history, street


def parse_infoset_to_network_input(infoset_str: str) -> Tuple[CanonicalCardsFromString, str]:
    """
    Parse infoset string and return inputs for DeepCFR network.

    This is the main function to use for getting network inputs from an infoset.

    Args:
        infoset_str: Infoset string from MCCFR

    Returns:
        Tuple of (canon_cards_object, action_history_string)
        These can be passed directly to DeepCFRModule.forward()
    """
    canon_cards_obj, action_history, street = parse_infoset_string(infoset_str)
    return canon_cards_obj, action_history


def batch_parse_infosets(infoset_strings: list) -> Tuple[list, list]:
    """
    Parse multiple infoset strings for batch processing.

    Args:
        infoset_strings: List of infoset strings

    Returns:
        Tuple of (list of canon_cards objects, list of action history strings)
    """
    canon_cards_list = []
    action_history_list = []

    for infoset_str in infoset_strings:
        canon_cards_obj, action_history = parse_infoset_to_network_input(
            infoset_str)
        canon_cards_list.append(canon_cards_obj)
        action_history_list.append(action_history)

    return canon_cards_list, action_history_list


def extract_infoset_features(infoset_str: str) -> Dict[str, any]:
    """
    Extract all features from an infoset string (for debugging/analysis).

    Args:
        infoset_str: Infoset string from MCCFR

    Returns:
        Dictionary with all parsed features
    """
    canon_cards_obj, action_history, street = parse_infoset_string(infoset_str)

    return {
        'street': street,
        'canonical_hand': canon_cards_obj.canonical_hand,
        'canonical_board': canon_cards_obj.canonical_board,
        'action_history': action_history,
        'hand_str': canon_cards_obj.get_hand_str(),
        'board_str': canon_cards_obj.get_board_str() if canon_cards_obj.canonical_board else '',
        'canon_cards_obj': canon_cards_obj
    }
