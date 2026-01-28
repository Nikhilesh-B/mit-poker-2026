''' Pokerbots Toss or Hold Em Training Gym '''

# %% Import Libraries

import sys, os
sys.path.append(os.getcwd())

import numpy as np
import json

from statistics import mean, stdev, StatisticsError
from itertools import chain

import engine_rl as Engine
import player_rl as TrainingBots

from skeleton.actions import FoldAction, CallAction, CheckAction, \
DiscardAction, RaiseAction
from skeleton.states import RoundState, GameState, TerminalState

import pkrbot
import poker_utils_rl
from poker_utils_rl import RANK_MAP, SUIT_MAP

import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, Dict, Discrete

import warnings
warnings.simplefilter('ignore')

# %% Variables

NUM_ROUNDS = 1000
STARTING_STACK = 400
BIG_BLIND = 2
SMALL_BLIND = 1

RAISE_POT_MULTI = [0.4, 0.8, 1.2]
ROUND_REWARD_SCALER = 0.05
GAME_OVER_REWARD_SCALER = 5

def keystoint(x):
    return {int(k): v for k, v in x.items()}
def valstoint(x):
    return {k: int(v) for k, v in x.items()}

with open('cards_decode.json', 'r') as f:
    CARDS_FROM_INT = json.load(f, object_hook=keystoint)

with open('cards_encode.json', 'r') as f:
    INT_FROM_CARDS = json.load(f, object_hook=valstoint)

del f


# %% Gym Environment

class TossHold(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self):
        # Continous Observation:
# TODO:     # Hand Eval: From 'pkrbot'
# TODO:     # Win Probability
# TODO:     # Pot Odds

        self.henry_rank_map = {
            "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
            "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

        self.Game = Engine.PokerGame()
        
        self.observation_space = Dict({
            'Discrete_Obs': MultiDiscrete(nvec=[
                2, 7, # SB/BB, Street
                15, 5, 15, 5, 15, 5, # Hole Cards
                15, 5, 15, 5, # Flop
                15, 5, 15, 5, # Discards
                15, 5, 15, 5, # Turn, River
                2, 2, 2, 2, # Legal Moves - Fold, Check, Call, Raise
                3, 3, 3, 3, 3, 3, 3, 3], # Hand Encoding
                dtype=np.int16),
            'Continuous_Obs': Box(
                # My Current Stack [0:400] / 400
                # Pot Stack [0:800] / 400
                # Current Round [0:1000] / 1000
                # Rolling Sum of Money Earned [-1000*400:1000*400] / 400
                low=np.array( [0.0, 0.0, 0.0, -1.0]),
                high=np.array([1.0, 2.0, 1.0,  1.0]), 
                dtype=np.float32)
            })
        
        # {0: Fold, 1: Check, 2: Call, 
        #  3: Raise 0.4*Pot, 4: Raise 0.8*Pot, 
        #  5: Raise 1.2*Pot, 6: All-In}
        self.action_space = Discrete(7)
        
        self.do_i_render_at_step = False
    
    def _get_obs(self):
        # ---- Discrete Observation ----
        discrete_obs = []
        
        # SB/BB
        if self.Game.sb == 'rl':
            discrete_obs.append(0)
        else:
            discrete_obs.append(1)
        
        # Street
        discrete_obs.append(self.Game.round_street)
        
        # Cards
        hole_flat, board_flat = self.get_visible_cards_encoding()
        discrete_obs += hole_flat
        discrete_obs += board_flat
        
        # Legal Moves
        legal_moves_enc = self.get_legal_moves_encoding()
        discrete_obs += legal_moves_enc
        
        # Hand Encodings
        hand_enc = poker_utils_rl.evaluate_poker_hands_list(
            self.Game.rl_cards, self.Game.board)
        # Clip size, shouldn't have impact.
        if len(hand_enc) > 8:
            print(f'Wrong Hand Encoding: {hand_enc}')
            hand_enc = hand_enc[:8]
        discrete_obs += hand_enc
        
        # ---- Continuous Observation ----
        cont_obs = []
        
        # My Stack
        my_stack = self.Game.rl_stack / STARTING_STACK
        cont_obs.append(my_stack)
        
        # Pot 
        pot = self.Game.pot / (2 * STARTING_STACK)
        cont_obs.append(pot)
        
        # Current Round
        curr_round = self.Game.round_no / NUM_ROUNDS
        cont_obs.append(curr_round)
        
        # Total Bankroll
        tot_bankroll = self.Game.rl_bankroll / (STARTING_STACK*NUM_ROUNDS)
        cont_obs.append(tot_bankroll)
        
        # ---- Combine ----
        combined_obs = {'Discrete_Obs': np.array(discrete_obs, 
                                                 dtype=np.int16), 
                        'Continuous_Obs': np.array(cont_obs, 
                                                   dtype=np.float32)}
        
        return combined_obs
    
    def _get_info(self):
        info_ = {'current_round': self.Game.round_no, 
                 'current_bankroll': self.Game.rl_bankroll, 
                 'hole_cards': self.Game.rl_cards, 
                 'board_cards': self.Game.board}
        
        return info_
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.Game.start_game()
        self.Game.start_round()
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def card_mapper(self, cards):
        mapped_cards = []
        for card in list(cards): 
            rank = RANK_MAP[card[0]]
            suit = SUIT_MAP[card[1]]
            mapped_cards.append([rank, suit])
        return mapped_cards
    
    def get_visible_cards_encoding(self):
        rl_cards = self.Game.rl_cards
        board = self.Game.board
        
        hole_cards_enc = self.card_mapper(rl_cards)
        board_cards_enc = self.card_mapper(board)
        
        # Padding
        while len(hole_cards_enc) < 3:
            hole_cards_enc.append([0, 0])
        while len(board_cards_enc) < 6:
            board_cards_enc.append([0, 0])
            
        # Flatten
        hole_cards_enc = list(chain.from_iterable(hole_cards_enc))
        board_cards_enc = list(chain.from_iterable(board_cards_enc))
        
        return hole_cards_enc, board_cards_enc
    
    def get_legal_moves_encoding(self):
        if self.Game.sb == 'rl':
            player_active = 0
        else:
            player_active = 1
        
        rs = self.Game.encode_round_state(player_active)
        legals = rs.legal_actions()
        allowed_actions = [x.__name__ for x in legals]
        
        fold_ = 1 if 'FoldAction' in allowed_actions else 0
        check_ = 1 if 'CheckAction' in allowed_actions else 0
        call_ = 1 if 'CallAction' in allowed_actions else 0
        raise_ = 1 if 'RaiseAction' in allowed_actions else 0
        legals_enc = [fold_, check_, call_, raise_]
        
        return legals_enc
    
    def handle_discard_action(self):
        # XXX: Copying PlayerHenry_v1 logic.
        my_cards = self.Game.rl_cards
        board_cards = self.Game.board
        
        hand_ranks = [self.henry_rank_map[c[0]] for c in my_cards]
        hand_suits = [c[1] for c in my_cards]
        board_ranks = [self.henry_rank_map[c[0]] for c in board_cards]
        board_suits = [c[1] for c in board_cards]

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
    
    def decode_action(self, rl_action):
        pot = self.Game.pot
        r1, r2, r3 = RAISE_POT_MULTI[0]*pot, RAISE_POT_MULTI[
            1]*pot, RAISE_POT_MULTI[2]*pot
        rl_pip = self.Game.rl_pips
        
        # XXX: For RaiseAction ?
        # The value that goes into the RaiseAction is how much 
        # in total we put into the pot.
        # So, amt would be pips + rX ? 
        
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
        
    def step(self, action):
        # Default
        reward = 0.0
        
        # If terminated, it will call self.reset() automatically.
        # game_over = terminated
        terminated, truncated = False, False
        
        opp_fold, rl_fold = False, False
        
        obs_prev = self._get_obs()
        info_prev = self._get_info()
        
        
        # Handle Discard - Assuming Opponent is logical.
        # XXX: Maybe we can stack Streets 2 and 3 with one if?
        if self.Game.round_street == 2:
            # BB Discards, SB Checks
            if self.Game.sb == 'oppo':
                discard_rl = self.handle_discard_action()
                self.Game.process_rl_move(discard_rl)
                self.Game.process_oppo_move()
            else:
                self.Game.process_oppo_move()
                self.Game.process_rl_move(CheckAction())
        if self.Game.round_street == 3:
            # SB Discards, BB Checks
            if self.Game.sb == 'oppo':
                self.Game.process_oppo_move()
                self.Game.process_rl_move(CheckAction())
            else:
                discard_rl = self.handle_discard_action()
                self.Game.process_rl_move(discard_rl)
                self.Game.process_oppo_move()
            
        # Opponent plays.
        while not self.Game.is_rl_turn():
            round_end, opp_fold = self.Game.process_oppo_move()
            
            # XXX: Should I start a new round here?
            if round_end:
                terminated = self.Game.end_round(opp_fold, rl_fold)
                reward += np.cbrt(self.Game.rl_bankroll_diff[-1]
                                  )*ROUND_REWARD_SCALER
                
                if terminated:
                    mean_win, std_win, sharpe = self.Game.end_game()
                    if mean_win < 0:
                        reward += -10
                    else:
                        reward += sharpe*GAME_OVER_REWARD_SCALER
                else:
                    self.Game.start_round()
        
        # RL's Turn
        if not terminated:
            # Here, process rl_action.
            rl_action_decoded = self.decode_action(action)
            round_end, rl_fold = self.Game.process_rl_move(rl_action_decoded)
            
            # XXX: Should I start a new round here?
            if round_end:
                terminated = self.Game.end_round(opp_fold, rl_fold)
                reward += np.cbrt(self.Game.rl_bankroll_diff[-1]
                                  )*ROUND_REWARD_SCALER
                
                if terminated:
                    mean_win, std_win, sharpe = self.Game.end_game()
                    if mean_win < 0:
                        reward += -10
                    else:
                        reward += sharpe*GAME_OVER_REWARD_SCALER
                else:
                    self.Game.start_round()
        
        # Return the next agent observation.
        if terminated or truncated:
            observation = obs_prev
            info = info_prev
        else:
            observation = self._get_obs()
            info = self._get_info()
        
        if self.do_i_render_at_step:
            self.render()

        return observation, reward, terminated, truncated, info
    
    def render(self):
        pass
        
    def close(self):
        pass
    