''' Pokerbots Toss or Hold Em Training Gym '''

# %% Import Libraries

import sys, os
sys.path.append(os.getcwd())

import numpy as np
from statistics import mean, stdev
from itertools import chain

import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, Dict

import engine_rl as Engine
from engine_rl import FoldAction, CallAction, CheckAction, \
RaiseAction, DiscardAction, TerminalState, GameState, RoundState

import pkrbot
import poker_utils
from poker_utils import RANK_MAP, SUIT_MAP

from player_rl import PlayerSkeleton, PlayerCeylan_v1, PlayerHenry_v1

NUM_ROUNDS = 1000
STARTING_STACK = 400
BIG_BLIND = 2
SMALL_BLIND = 1

import warnings
warnings.simplefilter('ignore')


# %% Gym Environment

class TossHold(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self):
        # XXX: Variable Opponent?
        self.Opp = PlayerSkeleton()
        
        self.start_stack = STARTING_STACK
        self.tot_rounds = NUM_ROUNDS
        
        self.render_mode = 'human'
        
        self.reset()
        
        # Discrete Observation:
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
            
        # XXX: I guess instead of Dict, we can use one Single Box ?
        # Or maybe not? How critical is it to have int/float seperation?
        self.observation_space = Dict({
            'Discrete_Obs': MultiDiscrete(nvec=[
                2, 7, 
                15, 5, 15, 5, 15, 5, 
                15, 5, 15, 5, 
                15, 5, 15, 5,
                15, 5, 
                15, 5, 
                3, 3, 3, 3, 3, 3, 3, 3, 
                2, 2, 2, 2], 
                dtype=np.int16),
            'Continuous_Obs': Box(
                low=np.array( [0.0, 0.0, 0.0, -1.0, -1.0]),
                high=np.array([1.0, 2.0, 1.0,  1.0,  1.0]), 
                dtype=np.float32)
            })
        
        
        # First Number is Fold, Check, Call, Raise, Discard
        # Second Number - if raise 1-10 between min/max raise bound.
        # Third Number is Discard Card Index
        self.action_space = MultiDiscrete(nvec=[5, 10, 3], dtype=np.int16)
    
    def _get_obs(self):
        round_state = self.Game.current_round_state
        
        # ---- Discrete Observation ----
        discrete_obs = []
        
        # SB/BB, Street
        discrete_obs.append(self.rl_pos)
        discrete_obs.append(round_state.street)
        
        # Update Card Info
        self.get_visible_cards_info()
        
        # Flatten Cards
        hole_flat = list(chain.from_iterable(self.hole_cards))
        board_flat = list(chain.from_iterable(self.board_cards))
        discrete_obs += hole_flat
        discrete_obs += board_flat
        
        # Hand Encodings
        hand_enc = poker_utils.evaluate_poker_hands_list(
            self.hole_cards_mini, self.board_cards_mini)
        discrete_obs += hand_enc
        
        # Legal Moves
        moves = round_state.legal_actions()
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
            r_bound = max_b/STARTING_STACK
        cont_obs.append(r_bound)
        
        # ---- Combine ----
        combined_obs = {'Discrete_Obs': np.array(discrete_obs, 
                                                 dtype=np.int16), 
                        'Continuous_Obs': np.array(cont_obs, 
                                                   dtype=np.float32)}
        
        return combined_obs
        
    
    def _get_info(self):
        info_ = {'current_round': self.Game.current_round, 
                 'current_bankroll': self.Game.P2_bankroll, 
                 'hole_cards': self.hole_cards, 
                 'board_cards': self.board_cards}
        
        return info_
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.bankroll = [0]
        
        # TODO: Add variable opponent here.
        self.Game = Engine.Game(self.Opp)
        self.start_new_round()
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def start_new_round(self):
        # Handle multiple round starts.
        while True:
            self.Game.start_new_round()
            # 0 is SB, 1 is BB.
            self.rl_pos = 0 if self.Game.sb == 'rl' else 1
            self.get_visible_cards_info()
            
            # Run until RL's turn or Game Over.
            while True:
                game_move_code = self.Game.move_game_forward()
                
                # RL Turn, Exit Function
                if game_move_code == 1:
                    return
                
                # Opponent Folded
                elif game_move_code == -1:
                    self.Game.end_round(self.Game.current_round_state)
                    self.bankroll.append(self.Game.P2_bankroll)
                    # Game Over
                    if self.Game.check_game_over():
                        return
                    # Return to Outer Loop to Start a New Round
                    break
    
    def card_mapper(self, cards):
        mapped_cards = []
        for card in cards:
            rank = RANK_MAP[card.__str__()[0]]
            suit = SUIT_MAP[card.__str__()[1]]
            mapped_cards.append([rank, suit])
        return mapped_cards
    
    def get_visible_cards_info(self):
        round_state = self.Game.current_round_state
        
        self.hole_cards = self.card_mapper(round_state.hands[self.rl_pos])
        self.board_cards = self.card_mapper(round_state.board)
        
        # Ideally, we should keep the order of which the cards were initially 
        # distributed, i.e. discarded card's place should be replaced 
        # with [0, 0], but should be fine for now.
        
        self.hole_cards_mini = self.hole_cards[:]
        self.board_cards_mini = self.board_cards[:]
        
        # Padding
        while len(self.hole_cards) < 3:
            self.hole_cards.append([0, 0])
        
        while len(self.board_cards) < 6:
            self.board_cards.append([0, 0])
        
    
    def decode_action(self, rl_action):
        fccrd = rl_action[0]
        raise_multi = rl_action[1]
        discard_idx = rl_action[2]
        
        if fccrd == 0:
            return FoldAction()
        elif fccrd == 1:
            return CheckAction()
        elif fccrd == 2:
            return CallAction()
        elif fccrd == 3:
            min_raise, max_raise = self.Game.current_round_state.raise_bounds()
            raise_amt = min_raise + (raise_multi/9)*(max_raise - min_raise)
            return RaiseAction(round(raise_amt))
        else:
            return DiscardAction(discard_idx)
        
    
    def step(self, action):
        # RL performs an action, then the opponent performs an action.
        # Observation comes after opponent's action is completed.
        if self.Game.check_game_over():
            raise RuntimeError("step() called after episode termination.")
        
        # Default
        reward = 0
        
        legal_actions_this_turn = self.Game.current_round_state.legal_actions()
        rl_agent_action = self.decode_action(action)
        
        # Illegal Action, Gets Negative Reward
        # Act like engine: Check if possible, Fold otherwise
        if type(rl_agent_action) not in legal_actions_this_turn:
            if CheckAction in legal_actions_this_turn:
                self.Game.process_rl_train_action(CheckAction())
            else:
                self.Game.process_rl_train_action(FoldAction())
            reward = -0.1
        else:
            self.Game.process_rl_train_action(rl_agent_action)
        
        
        # While not our turn, progress game forward.
        while True:
            game_move_code = self.Game.move_game_forward()
            
            # RL Agent's Turn
            if game_move_code == 1:
                break
            
            # Round Over
            elif game_move_code == -1:
                # Update bankroll, get reward.
                self.Game.end_round(self.Game.current_round_state)
                self.bankroll.append(self.Game.P2_bankroll)
                # Reward is change in bankroll, scaled by starting stack.
                reward += (self.bankroll[-1] - self.bankroll[-2]
                           )/STARTING_STACK
                # Start new round.
                self.start_new_round()
                break
        
        self.get_visible_cards_info()
        
        # Game end at 1000 rounds.
        if self.Game.check_game_over():
            terminated = True
        else:
            terminated = False

        # Might have the timer here.
        truncated = False
        
        
        # The game result after actual 1000 rounds is the most important
        # part for us. And as long as it's greater than 0, we win.
        # Should still need to scale by 'win consistency' etc.
        # which we'll be using Sharpe Ratio.
        if terminated:
            try:
                sharpe = mean(self.bankroll)/stdev(self.bankroll)
            except ZeroDivisionError:
                sharpe = 0
            if self.bankroll[-1] > 0:
                reward += np.clip(10*sharpe, 0, 20)
            else:
                reward += -10
        
        # Return the next agent observation.
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def render(self):
        print('Round State: ', self.Game.current_round_state)
        
    def close(self):
        pass
    