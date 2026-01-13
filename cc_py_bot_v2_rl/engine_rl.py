'''
6.9630 MIT POKERBOTS GAME ENGINE for RL Training
'''

from collections import namedtuple
import math

import pkrbot
import sys
import os

sys.path.append(os.getcwd())
from config_rl import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND


DiscardAction = namedtuple('DiscardAction', ['card'])

FoldAction = namedtuple('FoldAction', [])
CallAction = namedtuple('CallAction', [])
CheckAction = namedtuple('CheckAction', [])
RaiseAction = namedtuple('RaiseAction', ['amount'])

TerminalState = namedtuple('TerminalState', ['deltas', 'previous_state'])

GameState = namedtuple('GameState', ['bankroll', 'game_clock', 'round_num'])


class RoundState(namedtuple('_RoundState', 
                            ['button', 'street', 'pips', 'stacks', 
                             'hands', 'deck', 'board', 'previous_state'])):
    '''
    Encodes the game tree for one round of poker.
    '''
    
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
        if self.street in (2, 3):
            return {DiscardAction} if active != self.street % 2 else {CheckAction}
        if continue_cost == 0:
            bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
            return {CheckAction, FoldAction} if bets_forbidden else {
                CheckAction, RaiseAction, FoldAction}
        raises_forbidden = (continue_cost == self.stacks[
            active] or self.stacks[1-active] == 0)
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
        if self.street == 6:
            return self.showdown()
        elif self.street == 0:
            new_street = 2
            button = 1
            self.board.extend(self.deck.peek(new_street))
        elif self.street == 2:
            new_street = 3
            button = 0
        elif self.street == 3:
            new_street = 4
            button = 1
        else:
            new_street = self.street + 1
            button = 1
            self.board.append(self.deck.peek(new_street - 1)[new_street - 2])

        return RoundState(button, new_street, [0, 0], 
                          self.stacks, self.hands, 
                          self.deck, self.board, self)

    def proceed(self, action):
        active = self.button % 2
        if isinstance(action, DiscardAction):
            if len(self.hands[active]) != 0:
                self.board.append(self.hands[active].pop(action.card))
            state = RoundState((1 - active) % 2, self.street, 
                               self.pips, self.stacks, self.hands, 
                               self.deck, self.board, self)
            return state
        if isinstance(action, FoldAction):
            delta = self.get_delta((1 - active) % 2)
            return TerminalState([delta, -delta], self)
        if isinstance(action, CallAction):
            if self.button == 0:
                return RoundState(1, 0, [BIG_BLIND] * 2, [
                    STARTING_STACK - BIG_BLIND] * 2, self.hands, 
                    self.deck, self.board,self)
            new_pips = list(self.pips)
            new_stacks = list(self.stacks)
            contribution = new_pips[1-active] - new_pips[active]
            new_stacks[active] -= contribution
            new_pips[active] += contribution
            state = RoundState(self.button + 1, self.street, new_pips, 
                               new_stacks, self.hands, self.deck, 
                               self.board, self)
            return state.proceed_street()
        if isinstance(action, CheckAction):
            if (self.street == 0 and self.button > 0
                ) or self.button > 1 or self.street == 2 or self.street == 3:
                return self.proceed_street()
            return RoundState(self.button + 1, self.street, self.pips, 
                              self.stacks, self.hands, self.deck, 
                              self.board, self)
        new_pips = list(self.pips)
        new_stacks = list(self.stacks)
        contribution = action.amount - new_pips[active]
        new_stacks[active] -= contribution
        new_pips[active] += contribution
        return RoundState(self.button + 1, self.street, new_pips, 
                          new_stacks, self.hands, self.deck, self.board, self)


class Game():
    def __init__(self, P1):
        # P1 is the Opponent Bot: Bot that RL is training against.
        self.P1 = P1
        
        self.current_round = 0
        
        self.P1_bankroll = 0
        self.P2_bankroll = 0
        
    
    def send_round_state(self, round_state):
        return round_state
    
    def get_action_response(self, func):
        return 
    
    def run_round(self, round_num):
        '''
        Runs one round of poker.
        '''
        
        self.P1.handle_new_round()
        
        deck = pkrbot.Deck()
        deck.shuffle()
        hands = [deck.deal(3), deck.deal(3)]
        board = []
        
        button = round_num % 2
        
        if button == 0:
            pips = [SMALL_BLIND, BIG_BLIND]
            stacks = [STARTING_STACK - SMALL_BLIND, 
                      STARTING_STACK - BIG_BLIND]
        else:
            pips = [BIG_BLIND, SMALL_BLIND]
            stacks = [STARTING_STACK - BIG_BLIND, 
                      STARTING_STACK - SMALL_BLIND]
        
        round_state = RoundState(button, 0, pips, stacks, 
                                 hands, deck, board, None)
        
        # TODO: I think this needs to not be a loop.
        # But something that steps in time.
        while not isinstance(round_state, TerminalState):
            active = round_state.button % 2
            
            if active == 0:
                gs = GameState(self.P1_bankroll, 120, round_num)
                action = self.P1.get_action(gs, round_state, active)
            else:
                gs = GameState(self.P2_bankroll, 120, round_num)
                # TODO: Need to get this from RL training thing.
                action = self.P2.get_action(gs, round_state, active)
            
            round_state = round_state.proceed(action)
        
        self.P1_bankroll += round_state.deltas[0]
        self.P2_bankroll += round_state.deltas[1]
            
        self.P1.handle_round_over()

    def run(self):
        '''
        Runs one game of poker.
        '''
        for round_num in range(NUM_ROUNDS):
            self.run_round(round_num)


if __name__ == '__main__':
    # Game().run()
    pass
    