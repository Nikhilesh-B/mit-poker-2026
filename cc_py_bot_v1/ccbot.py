''' This class is submits actions in player.py. '''

from skeleton.actions import FoldAction, CallAction, CheckAction, \
RaiseAction, DiscardAction

import poker_utils
from poker_utils import RANK_MAP, SUIT_MAP


class MyBot():
    def __init__(self):
        pass
        
    def handle_new_round(self):
        pass
    
    def card_mapper(self, cards):
        mapped_cards = []
        for card in cards:
            rank = RANK_MAP[card[0]]
            suit = SUIT_MAP[card[1]]
            mapped_cards.append([rank, suit])
        return mapped_cards
    
    
    def get_action(self, game_state, round_state, active):
        # !!! Can only re-raise if both players can afford it.
        legal_actions = round_state.legal_actions()
        
        my_cards = self.card_mapper(round_state.hands[active])
        board_cards = self.card_mapper(round_state.board)
        
        poker_hands = poker_utils.evaluate_poker_hands(
            my_cards, board_cards)
        score = poker_utils.poker_hand_scorer(my_cards, board_cards)
        
        print({0:'SB', 1:'BB'}[active])
        print(my_cards, board_cards, round(score, 3))
        print(poker_hands)
        print(legal_actions)
        
        # Street is number of cards on the board.
        # When not discarding at street 2 or 3, 
        # only legal action is to check.
        if active == 0:
            if round_state.street == 2:
                return CheckAction()
            if round_state.street == 3:
                discard_i = self.discard_chooser(my_cards, board_cards)
                print(my_cards)
                return DiscardAction(discard_i)
        if active == 1:
            if round_state.street == 3:
                return CheckAction()
            if round_state.street == 2:
                discard_i = self.discard_chooser(my_cards, board_cards)
                print(my_cards)
                return DiscardAction(discard_i)
        
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1-active]
        
        if score < 3:
            return FoldAction()
        if score > 6:
            if RaiseAction in legal_actions:
                min_raise, max_raise = round_state.raise_bounds()
                multi = min(1, (score/20))/2
                raise_amt = round(min_raise+(max_raise-min_raise)*multi)
                print(min_raise, max_raise, raise_amt)
                return RaiseAction(raise_amt)
        
        # They distinguish between check and call.
        # If the opponent has not raised, 'Call' is illegal, need 'Check'.
        if CheckAction in legal_actions:
            return CheckAction()
        else:
            return CallAction()
    
    
    def discard_chooser(self, hand_cards, board_cards):
        # TODO: Currently discards the minimum.
        # Also check pair, flush, straight etc.
        min_rank, min_loc = 15, -1
        
        for i, (r, s) in enumerate(hand_cards):
            if r < min_rank:
                min_loc = i
                min_rank = r
                
        print(min_loc, min_rank)
        print('Now ret disc 1')
                
        return min_loc
            

        