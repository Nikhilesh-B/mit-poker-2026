'''
6.9630 MIT Pokerbots Custom Game Engine for RL Training
'''

# %% Import Libraries



import numpy as np
import time
_seed = int(time.time())
np.random.seed(_seed)
print(f'Engine np.random Seed: {_seed}\n')

import pkrbot

import math
from statistics import mean, stdev
import json

import sys, os
sys.path.append(os.getcwd())

from skeleton.actions import FoldAction, CallAction, CheckAction, \
DiscardAction, RaiseAction
from skeleton.states import RoundState, GameState, TerminalState

from player_rl import PlayerSkeleton, PlayerCeylan_v1, PlayerHenry_v1


# %% Variables

NUM_ROUNDS = 25 # 1000
STARTING_STACK = 400
BIG_BLIND = 2
SMALL_BLIND = 1
GAME_TIMEOUT = 120

def keystoint(x):
    return {int(k): v for k, v in x.items()}
def valstoint(x):
    return {k: int(v) for k, v in x.items()}

with open('cards_decode.json', 'r') as f:
    CARDS_FROM_INT = json.load(f, object_hook=keystoint)

with open('cards_encode.json', 'r') as f:
    INT_FROM_CARDS = json.load(f, object_hook=valstoint)

# Bot Class
OPPONENT1 = PlayerSkeleton()
OPPONENT2 = PlayerCeylan_v1()
OPPONENT3 = PlayerHenry_v1()

OPPONENT = OPPONENT3

del f

# %% Custom Game Engine

class PokerGame():
    def __init__(self):
        self.game_no = 0
        
    def get_final_rl_pot_win_multi(self):
        # Royal Flush: 10092544
        # High Card 7: 1048623
        # log10 is about between 6 and 7.
        score_rl = pkrbot.evaluate(self.board_lib + self.rl_cards_lib)
        score_opp = pkrbot.evaluate(self.board_lib + self.opp_cards_lib)
        
        # Tie, split pot.
        if score_rl == score_opp:
            return 0
        # RL Wins
        elif score_rl > score_opp:
            return 1
        # Opponent Wins
        else:
            return -1
        
    def get_card_str_from_int(self, cards_list):
        cards_str = [CARDS_FROM_INT[x] for x in cards_list]
        return cards_str
    
    def get_card_lib_from_str(self, cards_str):
        cards_lib = [pkrbot.Card(x) for x in cards_str]
        return cards_lib
    
    def deal_cards(self):
        cards_round = np.random.choice(52, size=10, replace=False)
        
        self.rl_cards = self.get_card_str_from_int(cards_round[0:3])
        self.rl_cards_lib = self.get_card_lib_from_str(self.rl_cards)
        
        self.opp_cards = self.get_card_str_from_int(cards_round[3:6])
        self.opp_cards_lib = self.get_card_lib_from_str(self.opp_cards)
        
        self.board = []
        self.board_lib = self.get_card_lib_from_str(self.board)
        
        self.flop = self.get_card_str_from_int(cards_round[6:8])
        self.flop_lib = self.get_card_lib_from_str(self.flop)
        
        self.turn = self.get_card_str_from_int(cards_round[8:9])
        self.turn_lib = self.get_card_lib_from_str(self.turn)
        
        self.river = self.get_card_str_from_int(cards_round[9:10])
        self.river_lib = self.get_card_lib_from_str(self.river)
        
    
    def handle_action(self, round_state, action, player='oppo'):
        allowed_actions = [x.__name__ for x in 
                           round_state.legal_actions()]
        action_name = type(action).__name__
        
        if action_name in allowed_actions:
            return action
        else:
            if 'CheckAction' in allowed_actions:
                return CheckAction()
            else:
                return FoldAction()
    
    def process_oppo_move(self):
        prev_street = self.round_street
        
        # XXX: Multiple-raise button-value gets lost here.
        if self.sb == 'oppo':
            oppo_active = 0
        else:
            oppo_active = 1
        
        rs_opp = self.encode_round_state(oppo_active)
        gs_opp = self.encode_game_state('oppo')
        
        action_opp = self.Opp.get_action(gs_opp, rs_opp, oppo_active)
        action_opp_legal = self.handle_action(rs_opp, action_opp, 'oppo')
        action_name = type(action_opp_legal).__name__
        
        print('Action Oppo:', action_opp)
        print('Action Oppo Legal:', action_opp_legal)
        
        round_end, oppo_fold = False, False
        
        if action_name == 'FoldAction':
            round_end, oppo_fold = True, True
            
        elif action_name == 'DiscardAction':
            # Doesn't advance street, b/c the other player needs to check.
            discard_idx = action_opp_legal.card
            discarded_card = self.opp_cards[discard_idx]
            
            self.board.append(discarded_card)
            self.board_lib = self.get_card_lib_from_str(self.board)
            
            self.opp_cards.remove(discarded_card)
            self.opp_cards_lib = self.get_card_lib_from_str(self.opp_cards)
            
            self.player_turn = (self.player_turn + 1) % 2
            
        elif action_name == 'CheckAction':
            if self.round_street == 0:
                # Can only check here if we're BB, and SB has called.
                # Street advances.
                self.player_turn = 1 # BB's turn.
                self.round_street = 2
                self.board += self.flop[:]
                self.board_lib += self.flop_lib[:]
            elif self.round_street == 6:
                self.consec_check += 1
                # Showdown if double check.
                if self.consec_check >= 2:
                    round_end, oppo_fold = True, False
                else:
                    self.player_turn = (self.player_turn + 1) % 2
            # For Street = 2, BB discards and SB checks.
            elif self.round_street == 2:
                # Can only check here if we're SB. The street advances.
                self.round_street += 1
                self.player_turn = 0 # SB's turn to discard.
            # For Street = 3, SB discards and BB checks.
            elif self.round_street == 3:
                # Can only check here if we're BB. The street advances.
                self.round_street += 1
                self.player_turn = 1 # BB's turn pre-turn card reveal.
            else:
                self.consec_check += 1
                # Street advances if double check.
                if self.consec_check >= 2:
                    self.player_turn = 1 # BB's turn.
                    self.round_street += 1
                    
                    if self.round_street == 5:
                        self.board += self.turn[:]
                        self.board_lib += self.turn_lib[:]
                    elif self.round_street == 6:
                        self.board += self.river[:]
                        self.board_lib += self.river_lib[:]
                    else:
                        raise ValueError('Invalid Check Street - Opp.')
                # RL's turn.
                else:
                    self.player_turn = (self.player_turn + 1) % 2
            
        # Can only call if the other opponent has raised.
        elif action_name == 'CallAction':
            call_amt = self.rl_pips - self.oppo_pips
            
            self.oppo_stack -= call_amt
            self.oppo_pips += call_amt
            self.pot += call_amt
            
            if self.round_street == 0:
                # SB called BB, BB can raise.
                if ((self.sb == 'oppo') and (not self.sb_called_bb_t0)):
                    # SB called BB, BB can raise.
                    self.player_turn = (self.player_turn + 1) % 2
                    self.sb_called_bb_t0 = True
                # Regular Call.
                else:
                    self.player_turn = 1 # BB is next.
                    self.round_street = 2
                    self.board += self.flop[:]
                    self.board_lib += self.flop_lib[:]
            # All other street calls always advance.
            else:
                # Showdown.
                if self.round_street == 6:
                    round_end, oppo_fold = True, False
                # Advance street.
                else:
                    self.player_turn = 1 # BB's turn.
                    self.round_street += 1
                    if self.round_street == 5:
                        self.board += self.turn[:]
                        self.board_lib += self.turn_lib[:]
                    elif self.round_street == 6:
                        self.board += self.river[:]
                        self.board_lib += self.river_lib[:]
                    else:
                        raise ValueError('Invalid Call Street - Opp.')
            
        elif action_name == 'RaiseAction':
            # Raise is Raise to, not additional amount.
            # Raise doesn't advance street.
            amt = action_opp_legal.amount
            # Check Bounds
            lower_bound, upper_bound = rs_opp.raise_bounds()
            amt = max(lower_bound, min(amt, upper_bound))
            amt_incr = amt - self.oppo_pips
            
            self.oppo_stack -= amt_incr
            self.oppo_pips += amt_incr
            self.pot += amt_incr
            
            self.player_turn = (self.player_turn + 1) % 2
        else:
            raise ValueError('Invalid Action Type - Opponent!')
        
        self.prev_state = rs_opp
        
        # Reset Check counter.
        if prev_street != self.round_street:
            self.consec_check = 0
        
        return round_end, oppo_fold
    
    
    def process_rl_move(self, rl_action):
        prev_street = self.round_street
        
        if self.sb == 'rl':
            rl_active = 0
        else:
            rl_active = 1
            
        rs_rl = self.encode_round_state(rl_active)
        action_rl_legal = self.handle_action(rs_rl, rl_action, 'rl')
        action_name = type(action_rl_legal).__name__
        
        print('Action RL:', rl_action)
        print('Action RL Legal:', action_rl_legal)
        
        round_end, rl_fold = False, False
        
        if action_name == 'FoldAction':
            round_end, rl_fold = True, True
            
        elif action_name == 'DiscardAction':
            # Doesn't advance street, b/c the other player needs to check.
            discard_idx = action_rl_legal.card
            discarded_card = self.rl_cards[discard_idx]
            
            self.board.append(discarded_card)
            self.board_lib = self.get_card_lib_from_str(self.board)
            
            self.rl_cards.remove(discarded_card)
            self.rl_cards_lib = self.get_card_lib_from_str(self.rl_cards)
            
            self.player_turn = (self.player_turn + 1) % 2
            
        elif action_name == 'CheckAction':
            if self.round_street == 0:
                # Can only check here if we're BB, and SB has called.
                # Street advances.
                self.player_turn = 1 # BB's turn.
                self.round_street = 2
                self.board += self.flop[:]
                self.board_lib += self.flop_lib[:]
            elif self.round_street == 6:
                self.consec_check += 1
                # Showdown if double check.
                if self.consec_check >= 2:
                    round_end, rl_fold = True, False
                else:
                    self.player_turn = (self.player_turn + 1) % 2
            # For Street = 2, BB discards and SB checks.
            elif self.round_street == 2:
                # Can only check here if we're SB. The street advances.
                self.round_street += 1
                self.player_turn = 0 # SB's turn to discard.
            # For Street = 3, SB discards and BB checks.
            elif self.round_street == 3:
                # Can only check here if we're BB. The street advances.
                self.round_street += 1
                self.player_turn = 1 # BB's turn pre-turn card reveal.
            else:
                self.consec_check += 1
                # Street advances if double check.
                if self.consec_check >= 2:
                    self.player_turn = 1 # BB's turn.
                    self.round_street += 1
                    
                    if self.round_street == 5:
                        self.board += self.turn[:]
                        self.board_lib += self.turn_lib[:]
                    elif self.round_street == 6:
                        self.board += self.river[:]
                        self.board_lib += self.river_lib[:]
                    else:
                        raise ValueError('Invalid Check Street - RL.')
                # Oppo's turn.
                else:
                    self.player_turn = (self.player_turn + 1) % 2
            
        # Can only call if the other opponent has raised.
        elif action_name == 'CallAction':
            call_amt = self.oppo_pips - self.rl_pips
            
            self.rl_stack -= call_amt
            self.rl_pips += call_amt
            self.pot += call_amt
            
            if self.round_street == 0:
                # SB called BB, BB can raise.
                if ((self.sb == 'rl') and (not self.sb_called_bb_t0)):
                    # SB called BB, BB can raise.
                    self.player_turn = (self.player_turn + 1) % 2
                    self.sb_called_bb_t0 = True
                # Regular Call.
                else:
                    self.player_turn = 1 # BB is next.
                    self.round_street = 2
                    self.board += self.flop[:]
                    self.board_lib += self.flop_lib[:]
            # All other street calls always advance.
            else:
                # Showdown.
                if self.round_street == 6:
                    round_end, rl_fold = True, False
                # Advance street.
                else:
                    self.player_turn = 1 # BB's turn.
                    self.round_street += 1
                    if self.round_street == 5:
                        self.board += self.turn[:]
                        self.board_lib += self.turn_lib[:]
                    elif self.round_street == 6:
                        self.board += self.river[:]
                        self.board_lib += self.river_lib[:]
                    else:
                        raise ValueError('Invalid Call Street - RL.')
            
        elif action_name == 'RaiseAction':
            # Raise doesn't advance street.
            amt = action_rl_legal.amount
            # Check Bounds
            lower_bound, upper_bound = rs_rl.raise_bounds()
            amt = max(lower_bound, min(amt, upper_bound))
            amt_incr = amt - self.rl_pips
            
            self.rl_stack -= amt_incr
            self.rl_pips += amt_incr
            self.pot += amt_incr
            
            self.player_turn = (self.player_turn + 1) % 2
        else:
            raise ValueError('Invalid Action Type - RL!')
        
        self.prev_state = rs_rl
        
        # Reset Check counter.
        if prev_street != self.round_street:
            self.consec_check = 0
        
        return round_end, rl_fold
        
        
    def start_game(self, Opp=OPPONENT):
        # Reset Opponent
        self.Opp = Opp
        
        self.round_no = 0
        
        self.rl_bankroll = 0
        self.opp_bankroll = 0
        
        self.rl_bankroll_rolling = [0]
        self.rl_bankroll_diff = []
        
        self.game_start_time = time.perf_counter()
        
        
    def start_round(self):
        sb_bb_key = (self.game_no + self.round_no) % 2
        self.sb_called_bb_t0 = False
        self.consec_check = 0
        
        # Set round info based on who's SB/BB.
        if sb_bb_key == 0:
            self.sb = 'oppo'
            self.rl_pips = BIG_BLIND
            self.rl_stack = STARTING_STACK - BIG_BLIND
            self.oppo_pips = SMALL_BLIND
            self.oppo_stack = STARTING_STACK - SMALL_BLIND
            oppo_active = 0
        else:
            self.sb = 'rl'
            self.rl_pips = SMALL_BLIND
            self.rl_stack = STARTING_STACK - SMALL_BLIND
            self.oppo_pips = BIG_BLIND
            self.oppo_stack = STARTING_STACK - BIG_BLIND
            oppo_active = 1
             
        self.pot = SMALL_BLIND + BIG_BLIND
        
        self.deal_cards()
        self.round_street = 0
        self.prev_state = None
        
        # BB: 1, SB: 0
        self.player_turn = 0
        
        # Opponent Handle New Round
        rs_opp = self.encode_round_state(oppo_active)
        gs_opp = self.encode_game_state('oppo')
        
        try:
            self.Opp.handle_new_round(gs_opp, rs_opp, oppo_active)
        except Exception as e:
            print(f'Opp handle_new_round Exception: {e}')
    
        
    def end_round(self, opp_fold=False, rl_fold=False):
        self.round_no += 1
        
        # Need to get the differential win amount 
        # vs starting stack for calculations.
        if opp_fold:
            rl_delta = self.oppo_pips
            opp_delta = -self.oppo_pips
        elif rl_fold:
            rl_delta = -self.rl_pips
            opp_delta = self.rl_pips
        else:
            rl_win_multi = self.get_final_rl_pot_win_multi()
            rl_delta = round(rl_win_multi*self.pot/2)
            opp_delta = -rl_delta
            
        self.rl_bankroll += rl_delta
        self.opp_bankroll += opp_delta
        
        self.rl_bankroll_rolling.append(self.rl_bankroll)
        self.rl_bankroll_diff.append(self.rl_bankroll_rolling[
            -1] - self.rl_bankroll_rolling[-2])
        
        if self.sb == 'oppo':
            deltas = [opp_delta, rl_delta]
            oppo_active = 0
        else:
            deltas = [rl_delta, opp_delta]
            oppo_active = 1
        
        # Opponent Handle Round Over
        previous_state = self.prev_state
        ts_opp = TerminalState(deltas, previous_state)
        gs_opp = self.encode_game_state('oppo')
        
        print()
        print(self.rl_pips, self.oppo_pips)
        print(f'PnL RL: {rl_delta} - PnL Oppo: {opp_delta}')
        print('--------------------------\n')
        
        try:
            self.Opp.handle_round_over(gs_opp, ts_opp, oppo_active)
        except Exception as e:
            print(f'Opp handle_round_over Exception: {e}')
        
        game_end = False
        if self.round_no >= NUM_ROUNDS:
            game_end = True
            
        return game_end
    
    def end_game(self):
        self.game_no += 1
        
        mean_win = mean(self.rl_bankroll_diff)
        try:
            std_win = stdev(self.rl_bankroll_diff)
            sharpe = mean_win/std_win
        except Exception:
            sharpe = 0
        
        return mean_win, std_win, sharpe
        
    
    def encode_round_state(self, player_active):
        button = player_active
        street = self.round_street
        
        if self.sb == 'oppo':
            pips = [self.oppo_pips, self.rl_pips]
            stacks = [self.oppo_stack, self.rl_stack]
            hands = [self.opp_cards_lib, self.rl_cards_lib]
        else:
            pips = [self.rl_pips, self.oppo_pips]
            stacks = [self.rl_stack, self.oppo_stack]
            hands = [self.rl_cards_lib, self.opp_cards_lib]
            
        board = self.board_lib
        previous_state = self.prev_state
        
        return RoundState(button, street, pips, stacks, 
                          hands, board, previous_state)
    
    def encode_game_state(self, player_code):
        if player_code == 'oppo':
            bankroll = self.opp_bankroll
        else:
            bankroll = self.rl_bankroll
        
        game_clock = max(GAME_TIMEOUT - self.game_start_time, 1)
        round_num = self.round_no
        
        return GameState(bankroll, game_clock, round_num)
        
    def is_rl_turn(self):
        rl_turn = ((self.player_turn == 0) and (self.sb == 'rl')) or (
            (self.player_turn == 1) and (self.sb == 'oppo'))
        return rl_turn
    
    def move_game_forward(self, rl_action=None, Opp=OPPONENT):
        game_over, mean_win, std_win, sharpe = False, 0, 0, 0
        
        if self.is_rl_turn():
            round_over, rl_fold = self.process_rl_move(rl_action)
            opp_fold = False
        else:
            round_over, opp_fold = self.process_oppo_move()
            rl_fold = False
            
        if round_over:
            game_over = self.end_round(opp_fold, rl_fold)
            
            if game_over:
                mean_win, std_win, sharpe = self.end_game()
                self.start_game(Opp=Opp)
                
            self.start_round()
            
        return game_over, mean_win, std_win, sharpe


# %% Main for Test

if __name__ == '__main__':
    Game = PokerGame()
    Game.start_game()
    Game.start_round()
    
    game_over = False
    test_bot = PlayerSkeleton()
    
    while not game_over:
        print()
        print(Game.round_no, Game.round_street)
        print(Game.opp_cards, Game.rl_cards, Game.board)
        print(Game.sb)
        print(Game.rl_pips, Game.oppo_pips)
        print(Game.rl_stack, Game.oppo_stack) # !!! Wrong updates.
        print(Game.pot)
        rs = Game.encode_round_state(Game.player_turn)
        la = rs.legal_actions()
        print(la)
        print(rs.raise_bounds())
        
        
        # rs = Game.encode_round_state(Game.player_turn)
        # print(rs)
        # print(rs.legal_actions())
        
        if Game.is_rl_turn():
            # Pick random action.
            rs = Game.encode_round_state(Game.player_turn)
            gs = Game.encode_game_state('rl')
            
            # la = rs.legal_actions()
            
            # if DiscardAction in la:
            #     game_over, mean_win, std_win, sharpe = Game.move_game_forward(
            #         DiscardAction(0))
            # else:
            #     game_over, mean_win, std_win, sharpe = Game.move_game_forward(
            #         CallAction())
            
            # Using Skeleton Bot for Simulation/Testing.
            skelly_action = test_bot.get_action(gs, rs, Game.player_turn)
            game_over, mean_win, std_win, sharpe = Game.move_game_forward(
                skelly_action)
            
        else:
            game_over, mean_win, std_win, sharpe = Game.move_game_forward()
    
    print(f'Mean: {mean_win:.2f}\nStd: {std_win:.2f}\nSharpe: {sharpe:.2f}')
    