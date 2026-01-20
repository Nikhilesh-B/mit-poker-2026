''' Pokerbots Toss or Hold Em Training Gym '''

# %% Import Libraries

import sys
from config_rl import OPPONENT_PATH, NUM_ROUNDS, STARTING_STACK, \
BIG_BLIND, SMALL_BLIND

sys.path.append(OPPONENT_PATH)
from player import Player as Opponent

import poker_utils
import pkrbot
import engine_rl as Engine

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict

class TossHold(gym.Env):
    
    def __init__(self):
        self.Opp = Opponent()
        
        self.start_stack = STARTING_STACK
        self.tot_rounds = NUM_ROUNDS
        
        self.reset()
        
        # Discrete Observation:
            # Our Turn / Opponents Turn [0:1]
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
# TODO:     # Win Probability
# TODO:     # Pot Odds
            
        self.observation_space = Dict({
            'Discrete_Obs': MultiDiscrete(nvec=[
                2, 
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
        
        # TODO: Action Space can't be dict.
        # Raise needs to be extension of the others. 
        self.action_space = Dict({'FCCR': Discrete(4), 
                                  'Raise': Box(0.0, 1.0)})
    
    def _get_obs(self):
        # TODO
        pass
    
    def _get_info(self):
        info_ = {'current_round': self.current_round, 
                 'current_bankroll': self.current_bankroll, 
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
        self.hole_cards = []
        self.board_cards = []
        
        self.current_stack = self.start_stack
        self.current_pot = 0
        
    def encode_round_state(self):
        pass
    
    def decode_action(self):
        pass
    
    def step(self, action):
        # !!!: We do the action, then the opponent does the action.
        # Our observation comes after the opponent's ???
        
        
        
        
        # Game end at 1000 rounds.
        terminated = False

        # XXX: Might have the timer here.
        truncated = False

        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    