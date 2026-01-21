''' Pokerbots Toss or Hold Em Training Gym '''

# %% Import Libraries

import sys, os
sys.path.append(os.getcwd())

from config_rl import OPPONENT_PATH, NUM_ROUNDS, STARTING_STACK, \
BIG_BLIND, SMALL_BLIND

# TODO: Make opponent progressively better.
# For now, keeping it constant.
sys.path.append(OPPONENT_PATH)
from player import Player as Opponent

import poker_utils
from poker_utils import RANK_MAP, SUIT_MAP

import pkrbot
import engine_rl as Engine
from engine_rl import FoldAction, CallAction, CheckAction, \
RaiseAction, DiscardAction, TerminalState, GameState, RoundState

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict

# %% Gym Environment

class TossHold(gym.Env):
    
    def __init__(self):
        self.Opp = Opponent()
        
        self.start_stack = STARTING_STACK
        self.tot_rounds = NUM_ROUNDS
        
        self.reset()
        
        # Discrete Observation:
            # Our Turn / Opponents Turn [0:1]  
                # -> Removed, opponent automatically plays after us.
            # BB/SB [0:1]
            # Street [0:6]
            # Hole Card 1 [2:14, 1:4] # 0 if doesn't exist
            # Hole Card 2
            # Hole Card 3
            # Flop Card 1
            # Flop Card 2
            # Discarded Card 1
            # Discarded Card 2
            # Turn Card
            # River Card
            # Hand Encodings 9*[0:2]
            # Legal Moves: Fold, Check, Call, Raise 4*[0, 1]
        # Continous Observation:
            # My Current Stack [0:400] / 400
            # Pot Stack [0:800] / 400
            # Current Round [0:1000] / 1000
            # Rolling Sum of Money Earned [-1000*400:1000*400] / 400
            # Raise Bounds, -1 if illegal. +1 for All-in.
# TODO:     # Hand Eval: From 'pkrbot'
# TODO:     # Win Probability
# TODO:     # Pot Odds
            
        self.observation_space = Dict({
            'Discrete_Obs': MultiDiscrete(nvec=[
                2, 7, 
                15, 5, 15, 5, 15, 5, 
                15, 5, 15, 5, 
                15, 5, 15, 5,
                15, 5, 
                15, 5, 
                3, 3, 3, 3, 3, 3, 3, 3, 
                2, 2, 2, 2], dtype=np.int16),
            'Continuous_Obs': Box(
                low=np.array( [0.0, 0.0, 0.0, -1.0, -1.0]),
                high=np.array([1.0, 2.0, 0.0,  1.0,  1.0]))
            })
        
        
        # Action Space can't be Dict.
        # self.action_space = Dict({'FCCR': Discrete(4), 
        #                           'Raise': Box(0.0, 1.0)})
        
        # MultiDiscrete instead of Dict.
        # First Number is Fold, Check, Call, Raise
        # Second Number - if raise 1-10 between min/max raise bound.
        self.action_space = MultiDiscrete(nvec=[4, 10], dtype=np.int16)
    
    def _get_obs(self):
        round_state = self.Game.current_round_state
        
        # ---- Discrete Observation ----
        discrete_obs = []
        
        # SB/BB, Street
        discrete_obs.append(self.rl_pos)
        discrete_obs.append(round_state.street)
        
        # Update Card Info
        self.get_visible_cards_info()
        
        discrete_obs += self.hole_cards
        discrete_obs += self.board_cards
        
        # Hand Encodings
        hand_enc = poker_utils.evaluate_poker_hands_list(
            self.hole_cards_mini, self.board_cards_mini)
        discrete_obs += hand_enc
        
        # Legal Moves
        moves = round_state.legal_actions
        # if/else in order
        if FoldAction in moves:
            discrete_obs.append(1)
        else:
            discrete_obs.append(0)
        if CheckAction in moves:
            discrete_obs.append(1)
        else:
            discrete_obs.append(0)
        if CallAction in moves:
            discrete_obs.append(1)
        else:
            discrete_obs.append(0)
        if RaiseAction in moves:
            discrete_obs.append(1)
        else:
            discrete_obs.append(0)
        
        # ---- Continuous Observation ----
        cont_obs = []
        
        # My Stack
        my_stack = round_state.stacks[self.rl_pos] / STARTING_STACK
        cont_obs.append(my_stack)
        
        # Pot 
        pot = sum(round_state.pips) / (2*STARTING_STACK)
        cont_obs.append(pot)
        
        # Current Round
        curr_round = self.Game.current_round / NUM_ROUNDS
        cont_obs.append(curr_round)
        
        # Total Bankroll
        tot_bankroll = self.Game.P2_bankroll / STARTING_STACK
        cont_obs.append(tot_bankroll)
        
        # Raise Bounds
        if RaiseAction not in moves:
            r_bound = -1
        else:
            min_b, max_b = round_state.raise_bounds()
            r_bound = max_b/my_stack
        cont_obs.append(r_bound)
        
        # ---- Combine ----
        combined_obs = {'Discrete_Obs': discrete_obs, 
                        'Continuous_Obs': cont_obs}
        
        return combined_obs
        
    
    def _get_info(self):
        info_ = {'current_round': self.Game.current_round, 
                 'current_bankroll': self.Game.P2_bankroll, 
                 'hole_cards': self.hole_cards, 
                 'board_cards': self.board_cards}
        
        return info_
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.Game = Engine.Game(self.Opp)
        self.start_new_round()
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def start_new_round(self):
        self.Game.start_new_round()
        # 0 is SB, 1 is BB.
        self.rl_pos = 0 if self.Game.sb == 'rl' else 1
        self.get_visible_cards_info()
        
        while True:
            game_move_code = self.Game.move_game_forward()
            # RL Agent's Turn.
            if game_move_code == 1:
                break
            # !!! This shouldn't give '-1',
            # end of round here, but might need to check regardles.
    
    def card_mapper(self, cards):
        mapped_cards = []
        for card in cards:
            rank = RANK_MAP[card[0]]
            suit = SUIT_MAP[card[1]]
            mapped_cards.append([rank, suit])
        return mapped_cards
    
    def get_visible_cards_info(self):
        round_state = self.Game.current_round_state
        
        self.hole_cards = self.card_mapper(round_state.hands[self.rl_pos])
        self.board_cards = self.card_mapper(round_state.board)
        
        # XXX: Ideally, we should keep the order of which
        # the cards were initially distributed, 
        # i.e. discarded card's place should be replaced with [0, 0],
        # but should be fine for now.
        
        self.hole_cards_mini = self.hole_cards[:]
        self.board_cards_mini = self.board_cards[:]
        
        # Padding
        while len(self.hole_cards) < 3:
            self.hole_cards.append([0, 0])
        
        while len(self.board_cards) < 6:
            self.board_cards.append([0, 0])
        
    
    def decode_action(self, rl_action):
        fccr = rl_action[0]
        raise_multi = rl_action[1]
        
        if fccr == 0:
            return FoldAction()
        elif fccr == 1:
            return CheckAction()
        elif fccr == 2:
            return CallAction()
        else:
            min_raise, max_raise = self.Game.current_round_state.raise_bounds()
            raise_amt = min_raise + (raise_multi/9)*(max_raise - min_raise)
            return RaiseAction(round(raise_amt))
        
    
    def step(self, action):
        # !!!: We do the action, then the opponent does the action.
        # Our observation comes after the opponent's ???
        
        # !!! VERY IMPORTANT !!!
        # Currently doesn't do discard action by itself.
        # Need to manually implement that, but maybe we can 
        # add in as third field in MultiDiscrete.
        # If DiscardAction is legal, it need to be always played.
        
        # TODO: Need to get the timing right here.
        # B/c 
        rl_agent_action = self.decode_action(action)
        self.Game.process_rl_train_action(rl_agent_action)
        
        # While not our turn, progress game forward.
        while True:
            game_move_code = self.Game.move_game_forward()
            
            # RL Agent's Turn
            if game_move_code == 1:
                break
            # Round Over
            # TODO: Not sure how to handle this?
            elif game_move_code == -1:
                break
        
        # Game end at 1000 rounds.
        if self.Game.check_game_over():
            terminated = True
        else:
            terminated = False

        # XXX: Might have the timer here.
        truncated = False
        
        # TODO: Write the reward function.
        reward = 1 if terminated else 0

        # XXX: Check if this obs needs to before or after
        # RL makes its move.
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    