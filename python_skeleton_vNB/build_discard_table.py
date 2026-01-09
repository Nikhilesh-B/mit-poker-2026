"""
Generate a pickle discard lookup table for the bot.

Creates discard_table.pkl in the same directory as this script, containing:
  { "<sorted_hand>|<flop_signature>": discard_index }

Signatures match the runtime logic in player.py and handle boards of length 3 or 4.
"""

import argparse
import itertools
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence, Tuple, List
from tqdm import tqdm  # type: ignore
import pkrbot as pb

RANKS = "23456789TJQKA"
SUITS = "shdc"


def canonicalize_cards(cards: Sequence[str]) -> List[str]:
    """
    Canonicalize suits by first-seen ordering to exploit suit symmetry.
    """
    suit_map = {}
    next_suit = iter("shdc")
    out = []
    for c in cards:
        rank, suit = c[0], c[1]
        if suit not in suit_map:
            suit_map[suit] = next(next_suit)
        out.append(rank + suit_map[suit])
    return out


def hand_key(hand: Sequence[str], board: Sequence[str]) -> str:
    combined = list(hand) + list(board)
    canon = canonicalize_cards(combined)
    canon_hand = canon[: len(hand)]
    return "_".join(sorted(canon_hand))


def flop_signature(board, hand=()):
    """
    Board-aware signature; works for 3-card (first discard) or 4-card (second discard) boards.
    """
    if not board:
        return "empty"
    combined = list(hand) + list(board)
    canon = canonicalize_cards(combined)
    canon_board = canon[len(hand):]
    ranks = [c[0] for c in canon_board]
    suits = [c[1] for c in canon_board]
    rank_order = {r: i for i, r in enumerate(RANKS)}
    hi = max(ranks, key=lambda r: rank_order[r])
    lo = min(ranks, key=lambda r: rank_order[r])
    paired = int(len(set(ranks)) < len(ranks))
    suit_counts = Counter(suits).most_common()
    max_suit = suit_counts[0][1] if suit_counts else 1
    suit_tex = "rainbow" if max_suit == 1 else (
        "two" if max_suit == 2 else "mono")
    spread = rank_order[hi] - rank_order[lo]
    if spread <= 2:
        connect = "tight"
    elif spread <= 4:
        connect = "med"
    else:
        connect = "loose"
    return f"{suit_tex}|paired:{paired}|hi:{hi}|spread:{spread}|connect:{connect}"


def estimate_equity(kept, board, trials: int = 160) -> float:
    """Monte Carlo equity vs random opponent, with board fixed."""
    deck = pb.Deck()
    kept_cards = [pb.Card(c) for c in kept]
    board_cards = [pb.Card(c) for c in board]
    used = set(kept_cards + board_cards)
    deck.cards = [c for c in deck.cards if c not in used]
    wins = ties = 0
    need_board = 5 - len(board_cards)
    for _ in range(trials):
        deck.shuffle()
        opp = deck.deal(2)
        draw = deck.deal(need_board)
        final_board = board_cards + draw
        my_score = pb.evaluate(final_board + kept_cards)
        opp_score = pb.evaluate(final_board + opp)
        if my_score > opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1
        deck.cards.extend(opp + draw)
    return (wins + 0.5 * ties) / trials if trials else 0.0


def best_discard(hand3, board, trials: int = 160) -> int:
    best_idx, best_score = None, -1.0
    for di in range(3):
        kept = [c for i, c in enumerate(hand3) if i != di]
        score = estimate_equity(kept, board, trials)
        if score > best_score:
            best_score, best_idx = score, di
    return best_idx


def sample_iterable(seq: Sequence, k: int) -> Iterable:
    """Sample k items without replacement; if k >= len(seq), return seq."""
    if k >= len(seq):
        return seq
    return random.sample(seq, k)


def unique_canonical_hands(deck: Sequence[str]) -> List[Tuple[str, str, str]]:
    """
    Return one representative per suit-symmetric 3-card hand.
    """
    seen = set()
    unique = []
    for hand in itertools.combinations(deck, 3):
        canon = "_".join(sorted(canonicalize_cards(hand)))
        if canon in seen:
            continue
        seen.add(canon)
        unique.append(hand)
    return unique


def unique_canonical_boards(sampled_boards: Sequence[Sequence[str]]) -> List[Sequence[str]]:
    """
    Deduplicate boards by suit symmetry (without hand context).
    """
    seen = set()
    unique = []
    for board in sampled_boards:
        canon = "_".join(sorted(canonicalize_cards(board)))
        if canon in seen:
            continue
        seen.add(canon)
        unique.append(board)
    return unique


def build_table(board_samples: int, trials: int, seed: int, max_boards_per_hand: int, hand_samples: int) -> dict:
    random.seed(seed)
    table = {}
    deck = [r + s for r in RANKS for s in SUITS]

    # sample boards (both 3- and 4-card) to keep runtime reasonable
    all_boards = list(itertools.combinations(deck, 4))
    random.shuffle(all_boards)
    sampled = []
    for combo in tqdm(all_boards[: board_samples], desc="Sampling boards"):
        sampled.append(combo[:3])
        sampled.append(combo[:4])
    sampled = unique_canonical_boards(sampled)

    all_hands = unique_canonical_hands(deck)
    if hand_samples and hand_samples < len(all_hands):
        hands_iter = random.sample(all_hands, hand_samples)
        hand_total = hand_samples
    else:
        hands_iter = all_hands
        hand_total = len(all_hands)

    for hand3 in tqdm(hands_iter, total=hand_total, desc="Hands"):
        hand3 = list(hand3)
        used = set(hand3)
        per_hand_boards = sample_iterable(
            sampled, min(max_boards_per_hand, len(sampled)))
        for board in per_hand_boards:
            if any(c in used for c in board):
                continue
            sig = flop_signature(board, hand3)
            key = hand_key(hand3, board) + "|" + sig
            di = best_discard(hand3, list(board), trials=trials)
            table[key] = di
    return table


def main():
    parser = argparse.ArgumentParser(description="Build discard_table.pkl")
    parser.add_argument("--board-samples", type=int, default=5000,
                        help="number of 4-card board combos to sample (each yields 3- and 4-card boards)")
    parser.add_argument("--trials", type=int, default=120,
                        help="MC trials per discard candidate")
    parser.add_argument("--seed", type=int, default=7,
                        help="RNG seed for reproducibility")
    parser.add_argument(
        "--output", type=str, default="discard_table.pkl", help="output pickle filename")
    parser.add_argument(
        "--max-boards-per-hand", type=int, default=100, help="limit boards evaluated per hand to control runtime")
    parser.add_argument(
        "--hand-samples", type=int, default=0, help="optional limit on number of hands (C(52,3)=22100); set 0 to use all")
    args = parser.parse_args()

    table = build_table(args.board_samples, args.trials,
                        args.seed, args.max_boards_per_hand, args.hand_samples)
    out_path = Path(__file__).parent / args.output
    with out_path.open("wb") as f:
        pickle.dump(table, f)
    print(f"Wrote {out_path} entries={len(table)}")


if __name__ == "__main__":
    main()
