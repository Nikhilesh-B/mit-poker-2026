import numpy as np
from itertools import chain

from skeleton.actions import FoldAction, CallAction, CheckAction, \
RaiseAction, DiscardAction

import poker_utils_rl
from poker_utils_rl import RANK_MAP, SUIT_MAP

from stable_baselines3 import PPO

NUM_ROUNDS = 1000
STARTING_STACK = 400
BIG_BLIND = 2
SMALL_BLIND = 1
RAISE_POT_MULTI = [0.4, 0.8, 1.2]

class MyRLBot():
    def __init__(self):
        path1 = r"C:\Users\DELL\Desktop\MIT MFin\3_IAP_2026\6.9630\\"
        path2 = r"mit-poker-2026\cc_py_bot_v3_rl\ppobot_v0\\"
        path3 = r"best_model\best_model.zip"
        self.model = PPO.load(path1+path2+path3)
        
    def handle_new_round(self, game_state, round_state, active):
        pass
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass
    
    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        allowed_actions = [x.__name__ for x in legal_actions]
        
        # Should only be during Discard Phase.
        if len(legal_actions) == 1:
            if 'DiscardAction' in allowed_actions:
                action = self.handle_discard_action(round_state)
            elif 'CheckAction' in allowed_actions:
                action = CheckAction()
            else:
                action = CheckAction()
        else:
            obs = self._get_obs(game_state, round_state)
            action_rlbot, _ = self.model.predict(obs, deterministic=True)
            action = self.decode_action(action_rlbot, round_state, active)
            
            # TODO: Replace Check with Call and vice versa
            # if the bot plays illegally.
        
        return action
    
    def handle_discard_action(self, round_state):
        # Copying PlayerHenry_v1 logic.
        henry_rank_map = {
            "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
            "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
        
        # XXX: Is this even correct to use the button?
        my_cards = round_state.hands[round_state.button % 2]
        board_cards = round_state.board
        
        hand_ranks = [henry_rank_map[c.__str__()[0]] for c in my_cards]
        hand_suits = [c.__str__()[1] for c in my_cards]
        board_ranks = [henry_rank_map[c.__str__()[0]] for c in board_cards]
        board_suits = [c.__str__()[1] for c in board_cards]

        keep_values = []
        
        for i in range(3):
            val = 0.0
            r = hand_ranks[i]
            s = hand_suits[i]

            # Pair Bonus
            pair_count = hand_ranks.count(r)
            if pair_count == 2:
                val += 12.0
            elif pair_count == 3:
                val += 15.0

            # Board Pair Synergy
            board_pair_count = board_ranks.count(r)
            if board_pair_count >= 2:
                val += 10.0
            elif board_pair_count == 1:
                val += 4.0

            # Flush Potential
            my_suit_count = hand_suits.count(s)
            board_suit_count = board_suits.count(s)
            total_suit = my_suit_count + board_suit_count
            if total_suit >= 4:
                val += 8.0
            elif total_suit == 3:
                val += 4.0

            # Straight Potential
            all_ranks = hand_ranks + board_ranks
            if r == 14:
                all_ranks.append(1)
            unique_ranks = sorted(set(all_ranks))
            max_straight_draw = 0
            for base in unique_ranks:
                window = [x for x in unique_ranks if base <= x < base + 5]
                max_straight_draw = max(max_straight_draw, len(window))
            if max_straight_draw >= 4:
                contributes = False
                for base in unique_ranks:
                    if base <= r < base + 5:
                        contributes = True
                        break
                if r == 14 and (1 in unique_ranks or 2 in unique_ranks):
                    contributes = True
                if contributes:
                    val += 6.0

            # High Card Value
            if r == 14:
                val += 5.0
            elif r >= 12:
                val += 3.0
            elif r >= 10:
                val += 1.5

            keep_values.append(val)

        discard_idx = keep_values.index(min(keep_values))
        
        return DiscardAction(discard_idx)
        
    def _get_obs(self, game_state, round_state):
        active = round_state.button % 2
        
        # ---- Discrete Observation ----
        discrete_obs = []
        
        my_cards = [x.__str__() for x in round_state.hands[active]]
        board_cards = [x.__str__() for x in round_state.board]
        
        # SB/BB
        discrete_obs.append(active)
        # Street
        discrete_obs.append(round_state.street)
        
        # Cards
        hole_flat, board_flat = self.get_visible_cards_encoding(
            my_cards, board_cards)
        discrete_obs += hole_flat
        discrete_obs += board_flat
        
        # Legal Moves
        legal_moves_enc = self.get_legal_moves_encoding(round_state)
        discrete_obs += legal_moves_enc
        
        # Hand Encodings
        hand_enc = poker_utils_rl.evaluate_poker_hands_list(
            my_cards, board_cards)
        
        # Clip size, shouldn't have impact.
        if len(hand_enc) > 8:
            print(f'Wrong Hand Encoding: {hand_enc}')
            hand_enc = hand_enc[:8]
        discrete_obs += hand_enc
        
        # ---- Continuous Observation ----
        cont_obs = []
        
        # My Stack
        my_stack = round_state.stacks[active] / STARTING_STACK
        cont_obs.append(my_stack)
        
        # Pot 
        pot = sum(round_state.pips) / (2 * STARTING_STACK)
        cont_obs.append(pot)
        
        # Current Round
        curr_round = game_state.round_num / NUM_ROUNDS
        cont_obs.append(curr_round)
        
        # Total Bankroll
        tot_bankroll = game_state.bankroll / (STARTING_STACK*NUM_ROUNDS)
        cont_obs.append(tot_bankroll)
        
        
        # ---- Combine ----
        combined_obs = {'Discrete_Obs': np.array(discrete_obs, 
                                                 dtype=np.int16), 
                        'Continuous_Obs': np.array(cont_obs, 
                                                   dtype=np.float32)}
        
        return combined_obs

    def get_legal_moves_encoding(self, round_state):
        legals = round_state.legal_actions()
        allowed_actions = [x.__name__ for x in legals]
        
        fold_ = 1 if 'FoldAction' in allowed_actions else 0
        check_ = 1 if 'CheckAction' in allowed_actions else 0
        call_ = 1 if 'CallAction' in allowed_actions else 0
        raise_ = 1 if 'RaiseAction' in allowed_actions else 0
        legals_enc = [fold_, check_, call_, raise_]
        
        return legals_enc
    
    def get_visible_cards_encoding(self, my_cards, board_cards):
        hole_cards_enc = self.card_mapper(my_cards)
        board_cards_enc = self.card_mapper(board_cards)
        
        # Padding
        while len(hole_cards_enc) < 3:
            hole_cards_enc.append([0, 0])
        while len(board_cards_enc) < 6:
            board_cards_enc.append([0, 0])
            
        # Flatten
        hole_cards_enc = list(chain.from_iterable(hole_cards_enc))
        board_cards_enc = list(chain.from_iterable(board_cards_enc))
        
        return hole_cards_enc, board_cards_enc
    
    def card_mapper(self, cards):
        mapped_cards = []
        for card in list(cards): 
            rank = RANK_MAP[card[0]]
            suit = SUIT_MAP[card[1]]
            mapped_cards.append([rank, suit])
        return mapped_cards
    
    def decode_action(self, rl_action, round_state, active):
        pot = sum(round_state.pips)
        r1, r2, r3 = RAISE_POT_MULTI[0]*pot, RAISE_POT_MULTI[
            1]*pot, RAISE_POT_MULTI[2]*pot
        
        rl_pip = round_state.pips[active]
        
        if rl_action == 0:
            return FoldAction()
        elif rl_action == 1:
            return CheckAction()
        elif rl_action == 2:
            return CallAction()
        elif rl_action == 3:
            amt = round(rl_pip + r1)
            return RaiseAction(amt)
        elif rl_action == 4:
            amt = round(rl_pip + r2)
            return RaiseAction(amt)
        elif rl_action == 5:
            amt = round(rl_pip + r3)
            return RaiseAction(amt)
        # All-In
        else:
            return RaiseAction(STARTING_STACK)