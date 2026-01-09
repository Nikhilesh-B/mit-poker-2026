"""
Build equity lookup tables for Toss or Hold'em (3-card start, discard to board, 6-card final board).

Outputs a pickle mapping:
    "<canon_hand>|<canon_board>" -> equity (win + 0.5*tie vs random opp)

You can choose hand size (3 pre-discard or 2 post-discard), current board size (0,2,3,4,5,6),
and sampling parameters. Uses suit-canonicalization to shrink state space.

We simulate remaining board cards to a final 6-board-card layout, then score best 5-card hand
from (2 or 3) hole + 6 board versus a random opponent hand (2 cards) with the same board.
"""

import argparse
import itertools
import pickle
import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple, List

from tqdm import tqdm  # type: ignore
import pkrbot as pb

RANKS = "23456789TJQKA"
SUITS = "shdc"
FINAL_BOARD_CARDS = 6  # Toss or Hold'em has 6 board cards at showdown


def canonicalize_cards(cards: Sequence[str]) -> List[str]:
    """Canonicalize suits by first-seen ordering."""
    suit_map = {}
    next_suit = iter("shdc")
    out = []
    for c in cards:
        rank, suit = c[0], c[1]
        if suit not in suit_map:
            suit_map[suit] = next(next_suit)
        out.append(rank + suit_map[suit])
    return out


def hand_board_key(hand: Sequence[str], board: Sequence[str]) -> str:
    combined = list(hand) + list(board)
    canon = canonicalize_cards(combined)
    h = canon[: len(hand)]
    b = canon[len(hand):]
    return "_".join(sorted(h)) + "|" + "_".join(sorted(b))


def unique_canonical_hands(deck: Sequence[str], hand_size: int) -> List[Tuple[str, ...]]:
    seen = set()
    unique = []
    for hand in itertools.combinations(deck, hand_size):
        canon = "_".join(sorted(canonicalize_cards(hand)))
        if canon in seen:
            continue
        seen.add(canon)
        unique.append(hand)
    return unique


def unique_canonical_boards(deck: Sequence[str], board_cards: int, samples: int, exclude: Sequence[str]) -> List[Tuple[str, ...]]:
    all_boards = list(itertools.combinations(deck, board_cards))
    random.shuffle(all_boards)
    if samples and samples < len(all_boards):
        all_boards = all_boards[:samples]
    seen = set()
    unique = []
    for board in all_boards:
        if any(c in exclude for c in board):
            continue
        canon = "_".join(sorted(canonicalize_cards(board)))
        if canon in seen:
            continue
        seen.add(canon)
        unique.append(board)
    return unique


def best5_score(cards: Sequence[pb.Card]) -> int:
    """Return best 5-card score from a list of pb.Card."""
    best = 0
    for combo in itertools.combinations(cards, 5):
        score = pb.evaluate(list(combo))
        if score > best:
            best = score
    return best


def estimate_equity(hand: Sequence[str], board: Sequence[str], trials: int) -> float:
    """Monte Carlo equity vs random opponent, completing to 6 board cards; best 5-card evaluation."""
    deck = pb.Deck()
    hand_cards = [pb.Card(c) for c in hand]
    board_cards = [pb.Card(c) for c in board]
    used = set(hand_cards + board_cards)
    deck.cards = [c for c in deck.cards if c not in used]
    need_board = FINAL_BOARD_CARDS - len(board_cards)
    if need_board < 0:
        raise ValueError("Board has more than final 6 cards")
    wins = ties = 0
    for _ in range(trials):
        deck.shuffle()
        opp = deck.deal(2)
        draw = deck.deal(need_board)
        final_board = board_cards + draw
        my_score = best5_score(final_board + hand_cards)
        opp_score = best5_score(final_board + opp)
        if my_score > opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1
        deck.cards.extend(opp + draw)
    return (wins + 0.5 * ties) / trials if trials else 0.0


def build_table(hand_size: int, board_cards: int, hand_samples: int, board_samples: int, trials: int, seed: int) -> dict:
    random.seed(seed)
    deck = [r + s for r in RANKS for s in SUITS]
    hands = unique_canonical_hands(deck, hand_size)
    if hand_samples and hand_samples < len(hands):
        hands = random.sample(hands, hand_samples)

    table = {}
    for hand in tqdm(hands, desc="Hands"):
        exclude = set(hand)
        boards = unique_canonical_boards(
            deck, board_cards, board_samples, exclude)
        for board in boards:
            key = hand_board_key(hand, board)
            eq = estimate_equity(hand, board, trials)
            table[key] = eq
    return table


def main():
    parser = argparse.ArgumentParser(description="Build equity_table.pkl")
    parser.add_argument("--hand-size", type=int, default=3,
                        choices=[2, 3], help="cards in hand (2 after discard, 3 pre-discard)")
    parser.add_argument("--board-cards", type=int, default=0,
                        choices=[0, 2, 3, 4, 5, 6], help="known board cards at this stage (used if board-cards-list is not provided)")
    parser.add_argument("--board-cards-list", type=str, default="0,2,3,4,5,6",
                        help="comma-separated list of board sizes (e.g. 0,3,5) to build in one run; defaults to all stages")
    parser.add_argument("--hand-samples", type=int, default=0,
                        help="optional cap on canonical hands (0 = all canonical)")
    parser.add_argument("--board-samples", type=int, default=0,
                        help="optional cap on canonical boards (0 = all canonical at this size)")
    parser.add_argument("--trials", type=int, default=2,
                        help="MC trials per state")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument("--output", type=str, default=None,
                        help="output pickle filename; if multiple board sizes are built, defaults to equity_tables.pkl")
    args = parser.parse_args()

    # Determine which board sizes to build
    if args.board_cards_list:
        boards_to_build = [
            int(b) for b in args.board_cards_list.split(",") if b.strip()]
    else:
        boards_to_build = [args.board_cards]

    tables = {}
    for bc in boards_to_build:
        if bc not in (0, 2, 3, 4, 5, 6):
            raise SystemExit(f"Invalid board size {bc}; allowed: 0,2,3,4,5,6")
        table = build_table(args.hand_size, bc,
                            args.hand_samples, args.board_samples, args.trials, args.seed)
        tables[(args.hand_size, bc)] = table
        print(
            f"Built table hand={args.hand_size} board={bc} entries={len(table)}")

    # Decide output filename
    if len(tables) == 1 and not args.output:
        hand_size, bc = next(iter(tables.keys()))
        out_name = f"equity_table_h{hand_size}_b{bc}.pkl"
        to_dump = next(iter(tables.values()))
    else:
        out_name = args.output or "equity_tables.pkl"
        to_dump = tables

    out_path = Path(__file__).parent / out_name
    with out_path.open("wb") as f:
        pickle.dump(to_dump, f)
    print(f"Wrote {out_path} tables={len(tables)}")


if __name__ == "__main__":
    main()
