'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import pickle
from collections import Counter
from pathlib import Path
from typing import Optional


class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.discard_table = {}
        self.equity_tables = {}
        self._load_discard_table()
        self._load_equity_tables()

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        # the total number of seconds your bot has left to play this game
        game_clock = game_state.game_clock
        round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        my_cards = round_state.hands[active]  # your cards
        big_blind = bool(active)  # True if you are the big blind
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        street = previous_state.street  # 0,2,3,4,5,6 representing when this round ended
        my_cards = previous_state.hands[active]  # your cards
        # opponent's cards or [] if not revealed
        opp_cards = previous_state.hands[1-active]
        pass

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        street = round_state.street
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.board  # the board cards
        equity = self._equity_lookup(my_cards, board_cards)
        # the number of chips you have contributed to the pot this round of betting
        my_pip = round_state.pips[active]
        # the number of chips your opponent has contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]
        # the number of chips you have remaining
        my_stack = round_state.stacks[active]
        # the number of chips your opponent has remaining
        opp_stack = round_state.stacks[1-active]
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot

        # the number of chips you have contributed to the pot
        my_contribution = STARTING_STACK - my_stack
        # the number of chips your opponent has contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack

        # Only use DiscardAction if it's in legal_actions (which already checks street)
        # legal_actions() returns DiscardAction only when street is 2 or 3
        if DiscardAction in legal_actions:
            return DiscardAction(self._choose_discard(my_cards, board_cards))

        # Equity-aware strategy
        equity = self._equity_lookup(my_cards, board_cards)
        pot_size = my_contribution + opp_contribution
        pot_odds_needed = continue_cost / \
            (pot_size + continue_cost) if continue_cost > 0 else 0

        if equity is not None:
            variance = equity * (1.0 - equity)
            ratio = equity / max(variance, 1e-4)

            # If equity does not clear pot odds and is low, fold/check
            if continue_cost > 0 and equity < pot_odds_needed and equity < 0.35:
                return FoldAction()

            # Aggressive when the ratio is high (stable edge)
            if ratio >= 2.0 and RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                target = min(max_raise, max(min_raise, my_pip + pot_size))
                return RaiseAction(target)

            # Pot control / cheap showdown
            if CheckAction in legal_actions:
                return CheckAction()
            if continue_cost == 0:
                return CheckAction()
            return CallAction()

        # Fallback heuristic if no equity available
        if RaiseAction in legal_actions:
            min_raise, _ = round_state.raise_bounds()
            if random.random() < 0.5:
                return RaiseAction(min_raise)
        if CheckAction in legal_actions:
            return CheckAction()
        if random.random() < 0.25:
            return FoldAction()
        return CallAction()

    def _choose_discard(self, hand, board):
        """
        Prefer precomputed table; fallback to heuristic (no Monte Carlo).
        """
        table_idx = self._lookup_discard_table(hand, board)
        if table_idx is not None:
            return table_idx
        return self._heuristic_discard(hand, board)

    def _lookup_discard_table(self, hand, board) -> Optional[int]:
        if not self.discard_table or not hand:
            return None
        key = self._hand_key(hand, board) + "|" + \
            self._flop_signature(board, hand)
        return self.discard_table.get(key)

    def _load_equity_tables(self) -> None:
        """
        Load precomputed equity tables.
        Supports either a combined pickle (equity_tables.pkl) with keys
        (hand_size, board_cards) -> table, or individual files named
        equity_table_h{hand}_b{board}.pkl.
        """
        tables_dir = Path(__file__).parent

        # Combined file
        combined = tables_dir / "equity_tables.pkl"
        if combined.exists():
            try:
                data = pickle.load(combined.open("rb"))
                if isinstance(data, dict):
                    for key, table in data.items():
                        if isinstance(key, tuple) and len(key) == 2:
                            self.equity_tables[key] = table
            except Exception:
                pass

        # Per-stage files
        for pkl in tables_dir.glob("equity_table_h*_b*.pkl"):
            parts = pkl.stem.split("_")
            try:
                hand_size = int(parts[2][1:])  # e.g. h2
                board_cards = int(parts[3][1:])  # e.g. b3
            except Exception:
                continue
            try:
                with pkl.open("rb") as f:
                    self.equity_tables[(hand_size, board_cards)
                                       ] = pickle.load(f)
            except Exception:
                pass

    def _equity_key(self, hand, board) -> str:
        """
        Match the key format used by build_equity_table.py.
        """
        combined = list(hand) + list(board)
        canon = self._canonicalize_cards(combined)
        h = canon[: len(hand)]
        b = canon[len(hand):]
        return "_".join(sorted(h)) + "|" + "_".join(sorted(b))

    def _equity_lookup(self, hand, board) -> Optional[float]:
        """
        Return precomputed win probability (vs random opponent) if available.
        """
        key = self._equity_key(hand, board)
        table = self.equity_tables.get((len(hand), len(board)))
        if not table:
            return None
        return table.get(key)

    def _load_discard_table(self) -> None:
        """
        Load precomputed discard decisions from a pickle file (discard_table.pkl).
        File is optional; absence just disables the table lookup.
        """
        table_path = Path(__file__).parent / "discard_table.pkl"
        if not table_path.exists():
            return
        try:
            with table_path.open("rb") as f:
                self.discard_table = pickle.load(f)
        except Exception:
            self.discard_table = {}

    def _hand_key(self, hand, board) -> str:
        combined = list(hand) + list(board)
        canon = self._canonicalize_cards(combined)
        canon_hand = canon[: len(hand)]
        return "_".join(sorted(canon_hand))

    def _flop_signature(self, board, hand=()) -> str:
        """
        Board-aware signature for discard lookup.
        Handles both 3-card (first discard) and 4-card (second discard) boards.
        """
        if not board:
            return "empty"
        combined = list(hand) + list(board)
        canon = self._canonicalize_cards(combined)
        canon_board = canon[len(hand):]
        ranks = [c[0] for c in canon_board]
        suits = [c[1] for c in canon_board]
        rank_order = {r: i for i, r in enumerate("23456789TJQKA")}
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

    def _canonicalize_cards(self, cards):
        """
        Canonicalize suits by order of appearance to exploit suit symmetry.
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

    def _heuristic_discard(self, hand, board):
        """
        Heuristic discard: keep pairs, keep suited/connected combos, drop weakest kicker.
        """
        rank_order = {r: i for i, r in enumerate("23456789TJQKA")}
        suits = [c[1] for c in hand]
        ranks = [c[0] for c in hand]

        # Keep a pair if present: discard the odd card out
        rank_counts = Counter(ranks)
        pair_rank = next(
            (r for r, cnt in rank_counts.items() if cnt >= 2), None)
        if pair_rank:
            candidates = [i for i, r in enumerate(ranks) if r != pair_rank]
            if candidates:
                return candidates[0]
            return min(range(len(hand)), key=lambda i: rank_order[ranks[i]])

        # Prefer suited combos (including board assistance)
        suit_counts = Counter(suits + [c[1] for c in board])
        best_suit = max(
            suit_counts, key=suit_counts.get) if suit_counts else None
        suited_idxs = [i for i, s in enumerate(suits) if s == best_suit]
        if len(suited_idxs) >= 2:
            off_suit = [i for i in range(len(hand)) if i not in suited_idxs]
            if off_suit:
                return off_suit[0]

        # Prefer connected/gapped combos: keep the two closest in rank
        values = [rank_order[r] for r in ranks]
        best_pair = (None, 99)
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                gap = abs(values[i] - values[j])
                if gap < best_pair[1]:
                    best_pair = ((i, j), gap)
        if best_pair[0]:
            keep = set(best_pair[0])
            discard_idx = next(i for i in range(len(hand)) if i not in keep)
            return discard_idx

        return min(range(len(hand)), key=lambda i: rank_order[ranks[i]])


if __name__ == '__main__':
    run_bot(Player(), parse_args())
