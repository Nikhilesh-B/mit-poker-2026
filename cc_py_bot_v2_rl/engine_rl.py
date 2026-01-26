'''
6.9630 MIT Pokerbots Game Engine for RL Training
'''

from collections import namedtuple
import math

import pkrbot

import sys, os
sys.path.append(os.getcwd())

NUM_ROUNDS = 1000
STARTING_STACK = 400
BIG_BLIND = 2
SMALL_BLIND = 1

FoldAction = namedtuple('FoldAction', [])
CallAction = namedtuple('CallAction', [])
CheckAction = namedtuple('CheckAction', [])
RaiseAction = namedtuple('RaiseAction', ['amount'])
DiscardAction = namedtuple('DiscardAction', ['card'])

TerminalState = namedtuple('TerminalState', ['deltas', 'previous_state'])
GameState = namedtuple('GameState', ['bankroll', 'game_clock', 'round_num'])

# Active: BB: 1, SB: 0

class RoundState(namedtuple('_RoundState', 
                            ['button', 'street', 'pips', 'stacks', 
                             'hands', 'deck', 'board', 'previous_state'])):
    '''
    Encodes the game tree for one round of poker.
    '''
    
    def sanitize(self):
        return RoundState(self.button, self.street, list(self.pips), 
                          list(self.stacks), self.hands, self.deck, 
                          self.board, self.previous_state)
    
    def get_delta(self, winner_index: int) -> int:
        assert winner_index in [0, 1, 2]
        delta = 0
        if winner_index == 2:
            assert(self.stacks[0] == self.stacks[1]) 
            delta = 0
        else:
            if winner_index == 0:
                delta = STARTING_STACK - self.stacks[1]
            else:
                delta = self.stacks[0] - STARTING_STACK
        if abs(delta - math.floor(delta)) > 1e-6:
            delta = math.floor(delta) if self.button % 2 == 0 else math.ceil(
                delta)
        return int(delta)

    def showdown(self) -> TerminalState:
        score0 = pkrbot.evaluate(self.board + self.hands[0])
        score1 = pkrbot.evaluate(self.board + self.hands[1])
        assert(self.stacks[0] == self.stacks[1])
        if score0 > score1:
            delta = self.get_delta(0)
        elif score0 < score1:
            delta = self.get_delta(1)
        else:
            delta = self.get_delta(2)
        
        return TerminalState([int(delta), -int(delta)], self)

    def legal_actions(self):
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        
        if self.street == 3:
            return {DiscardAction}
        
        if continue_cost == 0:
            bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
            return {CheckAction, FoldAction} if bets_forbidden else {
                CheckAction, RaiseAction, FoldAction}
        
        raises_forbidden = (continue_cost == self.stacks[active] or self.stacks[
            1-active] == 0)
        return {FoldAction, CallAction} if raises_forbidden else {
            FoldAction, CallAction, RaiseAction}

    def raise_bounds(self):
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        max_contribution = min(self.stacks[active], self.stacks[
            1-active] + continue_cost)
        min_contribution = min(max_contribution, continue_cost + max(
            continue_cost, BIG_BLIND))
        return (self.pips[active] + min_contribution, self.pips[
            active] + max_contribution)

    def proceed_street(self):
        new_board = list(self.board)
        new_hands = self.hands
        button = 1
        
        if self.street == 6:
            return self.showdown()
        
        if self.street == 0:
            new_street = 2
            new_board = self.deck.deal(3) 
    
        elif self.street == 2:
            new_street = 3
            button = 0
    
        elif self.street == 3:
            new_street = 4
            new_board.append(self.deck.deal(1)[0])
    
        elif self.street == 4:
            new_street = 5
            new_board.append(self.deck.deal(1)[0])
    
        elif self.street == 5:
            new_street = 6
            
        else:
            new_street = self.street + 1
    
        return RoundState(
            button, 
            new_street, 
            [0, 0],
            self.stacks, 
            new_hands, 
            self.deck, 
            tuple(new_board),
            self)
    
    def proceed(self, action):
        active = self.button % 2
        action_name = type(action).__name__
        
        if action_name == 'DiscardAction':
            new_hands = [list(self.hands[0]), list(self.hands[1])]
            new_board = list(self.board)
            
            if len(new_hands[active]) > 0:
                new_board.append(new_hands[active].pop(action.card))
            
            interim_state = RoundState(
                self.button + 1, 
                self.street, 
                self.pips, 
                self.stacks, 
                tuple(tuple(h) for h in new_hands), 
                self.deck, 
                tuple(new_board), 
                self)
            
            if self.button == 1:
                return interim_state.proceed_street() 
            return interim_state
        
        if action_name == 'CallAction':
            new_pips = list(self.pips)
            new_stacks = list(self.stacks)
            contribution = new_pips[1-active] - new_pips[active]
            new_stacks[active] -= contribution
            new_pips[active] += contribution
            
            if self.street == 0 and self.button == 0:
                return RoundState(1, self.street, new_pips, new_stacks, 
                                  self.hands, self.deck, self.board, self)
            
            state = RoundState(self.button + 1, self.street, new_pips, 
                               new_stacks, self.hands, self.deck, self.board, 
                               self)
            return state.proceed_street()
    
        if action_name == 'CheckAction':
            if (self.street == 0 and self.button == 1) or (
                    self.street != 0 and self.button == 2):
                return self.proceed_street()
            
            return RoundState(self.button + 1, self.street, self.pips, 
                              self.stacks, self.hands, self.deck, self.board, 
                              self)
        
        if action_name == 'FoldAction':
            delta = self.get_delta((1 - active) % 2)
            return TerminalState([delta, -delta], self)
    
        if action_name == 'RaiseAction':
            new_pips = list(self.pips)
            new_stacks = list(self.stacks)
            contribution = action.amount - new_pips[active]
            new_stacks[active] -= contribution
            new_pips[active] += contribution
            return RoundState(self.button + 1, self.street, new_pips, new_stacks, 
                              self.hands, self.deck, self.board, self)
    
        raise Exception(f'Invalid Action Type: {action}')

class Game():
    def __init__(self, P1):
        self.P1 = P1
        self.P1_bankroll = 0
        self.P2_bankroll = 0
        self.current_round = 0
    
    def start_new_round(self):
        if self.current_round % 2 == 0:
            self.sb = 'oppo'
        else:
            self.sb = 'rl'
            
        deck = pkrbot.Deck()
        deck.shuffle()
        
        hands = tuple(tuple(deck.deal(3)) for _ in range(2))
        board = () 
        pips = [SMALL_BLIND, BIG_BLIND]
        stacks = [STARTING_STACK - SMALL_BLIND, 
                  STARTING_STACK - BIG_BLIND]
        
        self.current_round_state = RoundState(0, 0, pips, stacks, 
                                              hands, deck, board, None)
        
        oppo_active_no = self.current_round % 2
        self.P1.handle_new_round(GameState(self.P1_bankroll, 120, 
                                           self.current_round), 
                                 self.current_round_state.sanitize(), 
                                 oppo_active_no)
    
    def end_round(self, round_state):
        if self.sb == 'oppo':
            self.P1_bankroll += round_state.deltas[0]
            self.P2_bankroll += round_state.deltas[1]
        else:
            self.P1_bankroll += round_state.deltas[1]
            self.P2_bankroll += round_state.deltas[0]
        
        oppo_active_no = self.current_round % 2
        self.P1.handle_round_over(GameState(self.P1_bankroll, 120,
                                            self.current_round), 
                                  self.current_round_state, oppo_active_no)
        
        self.current_round += 1
    
    
    def move_game_forward(self):
        if self.check_round_over():
            return -1
        
        active = self.current_round_state.button % 2
        if self.sb == 'oppo':
            if active == 0:
                self.process_opponent_action(active)
                return 0
            else:
                return 1
        else:
            if active == 0:
                return 1
            else:
                self.process_opponent_action(active)
                return 0
    
    def process_opponent_action(self, active):
        gs = GameState(self.P1_bankroll, 120, self.current_round)
        action = self.P1.get_action(gs, self.current_round_state.sanitize(), 
                                    active)
        self.current_round_state = self.current_round_state.proceed(action)
    
    def process_rl_train_action(self, rl_agent_action):
        self.current_round_state = self.current_round_state.proceed(
            rl_agent_action)
    
    def check_round_over(self):
        state_name = type(self.current_round_state).__name__
        if state_name == 'TerminalState':
            return True
        else:
            return False
    
    def check_game_over(self):
        if self.current_round >= NUM_ROUNDS:
            return True
        else:
            return False
    
    def get_round_state(self):
        return self.current_round_state
