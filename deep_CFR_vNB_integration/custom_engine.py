'''
Custom Engine for Deep CFR Training

This is a standalone version of the poker game engine for use with Deep CFR training.
It contains only the game state logic needed for MCCFR tree traversals.

Based on MIT Pokerbots 2026 game engine.
'''
from config import STARTING_STACK, BIG_BLIND, SMALL_BLIND
from collections import namedtuple
import math
import pkrbot  # import eval7, but better

# Action types
DiscardAction = namedtuple('DiscardAction', ['card'])
FoldAction = namedtuple('FoldAction', [])
CallAction = namedtuple('CallAction', [])
CheckAction = namedtuple('CheckAction', [])
RaiseAction = namedtuple('RaiseAction', ['amount'])

TerminalState = namedtuple('TerminalState', ['deltas', 'previous_state'])

STREET_NAMES = ['Flop', 'Discard 1', 'Discard 2', 'Turn', 'River']
DECODE = {'F': FoldAction, 'C': CallAction,
          'K': CheckAction, 'R': RaiseAction, 'D': DiscardAction}


def CCARDS(cards): return ','.join(map(str, cards))
def PCARDS(cards): return '[{}]'.format(' '.join(map(str, cards)))
def PVALUE(name, value): return ', {} ({})'.format(name, value)


class RoundState(namedtuple('_RoundState', ['button', 'street', 'pips', 'stacks', 'hands', 'deck', 'board', 'previous_state', 'action_taken'])):
    '''
    Encodes the game tree for one round of poker.

    The action_taken field stores the action that led to this state (for efficient history tracking).
    '''

    def get_delta(self, winner_index: int) -> int:
        '''Returns the delta for player A and -delta for player B.

        Args:
            winner_index (int): Index of the winning player. Must be 0 (player A),
                1 (player B), or 2 (split pot).

        Returns:
            int: The delta value for player A and -delta for player B.
        '''
        assert winner_index in [0, 1, 2]
        delta = 0
        if winner_index == 2:
            # Case of split pots
            # split pots only happen on the river + equal stacks
            assert (self.stacks[0] == self.stacks[1])
            delta = 0
        else:
            # Case of one player winning
            if winner_index == 0:
                delta = STARTING_STACK - self.stacks[1]
            else:
                delta = self.stacks[0] - STARTING_STACK

        # if delta is not an integer, round it down or up depending on who's in position
        if abs(delta - math.floor(delta)) > 1e-6:
            delta = math.floor(
                delta) if self.button % 2 == 0 else math.ceil(delta)
        return int(delta)

    def showdown(self) -> TerminalState:
        '''
        Compares the players' hands and computes the final payoffs at showdown.

        Evaluates both players' hands (hole cards + community cards) and determines
        the winner. The payoff (delta) is calculated based on:
        - The winner of the hand
        - The current pot size

        Returns:
            TerminalState: A terminal state object containing:
                - List of deltas (positive for winner, negative for loser)
                - Reference to the previous game state

        Note:
            This method assumes both players have equal stacks when reaching showdown,
            which is enforced by an assertion.
        '''
        score0 = pkrbot.evaluate(self.board + self.hands[0])
        score1 = pkrbot.evaluate(self.board + self.hands[1])
        assert (self.stacks[0] == self.stacks[1])
        if score0 > score1:
            delta = self.get_delta(0)
        elif score0 < score1:
            delta = self.get_delta(1)
        else:
            # split the pot
            delta = self.get_delta(2)

        return TerminalState([int(delta), -int(delta)], self)

    def legal_actions(self):
        '''
        Returns a set which corresponds to the active player's legal moves.
        '''
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        if self.street in (2, 3):
            return {DiscardAction} if active != self.street % 2 else {CheckAction}
        if continue_cost == 0:
            # we can only raise the stakes if both players can afford it
            bets_forbidden = (self.stacks[0] == 0 or self.stacks[1] == 0)
            return {CheckAction, FoldAction} if bets_forbidden else {CheckAction, RaiseAction, FoldAction}
        # continue_cost > 0
        # similarly, re-raising is only allowed if both players can afford it
        raises_forbidden = (
            continue_cost == self.stacks[active] or self.stacks[1-active] == 0)
        return {FoldAction, CallAction} if raises_forbidden else {FoldAction, CallAction, RaiseAction}

    def raise_bounds(self):
        '''
        Returns a tuple of the minimum and maximum legal raises.
        '''
        active = self.button % 2
        continue_cost = self.pips[1-active] - self.pips[active]
        max_contribution = min(
            self.stacks[active], self.stacks[1-active] + continue_cost)
        min_contribution = min(
            max_contribution, continue_cost + max(continue_cost, BIG_BLIND))
        return (self.pips[active] + min_contribution, self.pips[active] + max_contribution)

    def proceed_street(self):
        '''
        Resets the players' pips and advances the game tree to the next round of betting and updates the board state.

        possible streets: 0, 2, 3, 4, 5, 6
        '''
        # CRITICAL FIX: Deep copy mutable objects to prevent state mutation during MCCFR traversal
        new_hands = [list(h) for h in self.hands]
        new_board = list(self.board)

        # Put the board as peek deck of the street number and make sure board includes this peek + the players discarded cards after state
        if self.street == 6:
            return self.showdown()
        elif self.street == 0:
            new_street = 2
            button = 1  # Player B discards first, since they are out of position
            new_board.extend(self.deck.peek(new_street))
        elif self.street == 2:
            new_street = 3
            button = 0  # Player A discards second
        elif self.street == 3:
            new_street = 4
            button = 1  # Player B acts first after the discard phase
        else:
            new_street = self.street + 1
            button = 1
            new_board.append(self.deck.peek(new_street - 1)[new_street - 2])

        # Mark street transitions with None (not a player action, but a game phase change)
        return RoundState(button, new_street, [0, 0], self.stacks, new_hands, self.deck, new_board, self, None)

    def proceed(self, action):
        '''
        Advances the game tree by one action performed by the active player.

        Args:
            action: The action being performed. Must be one of:
                - DiscardAction: Player discards a card from their hand and adds it to the board
                - FoldAction: Player forfeits the hand
                - CallAction: Player matches the current bet
                - CheckAction: Player passes when no bet to match
                - RaiseAction: Player increases the current bet

        Returns:
            Either:
            - RoundState: The new state after the action is performed
            - TerminalState: If the action ends the hand (e.g., fold or final call)

        Note:
            The button value is incremented after each action to track whose turn it is.
            For DiscardAction, the card is added to the board and the hand is updated. Also, advances to the next street.
            For FoldAction, the inactive player is awarded the pot.
            For CallAction on button 0, both players post blinds.
            For CheckAction, advances to next street if both players have acted.
            For RaiseAction, updates pips and stacks based on raise amount.
        '''
        active = self.button % 2
        if isinstance(action, DiscardAction):
            # CRITICAL FIX: Deep copy mutable objects to prevent state mutation during MCCFR traversal
            # Without this, exploring multiple discard actions mutates the parent state!
            new_hands = [list(h) for h in self.hands]
            new_board = list(self.board)

            if len(new_hands[active]) != 0:
                new_board.append(new_hands[active].pop(action.card))

            # Use the NEW copies, not the original references
            state = RoundState((1 - active) % 2, self.street, self.pips,
                               self.stacks, new_hands, self.deck, new_board, self, action)
            return state
        if isinstance(action, FoldAction):
            # if active folds, the other player (1 - active) wins
            delta = self.get_delta((1 - active) % 2)
            return TerminalState([delta, -delta], self)
        if isinstance(action, CallAction):
            if self.button == 0:  # sb calls bb
                return RoundState(1, 0, [BIG_BLIND] * 2, [STARTING_STACK - BIG_BLIND] * 2, self.hands, self.deck, self.board, self, action)
            # both players acted
            new_pips = list(self.pips)
            new_stacks = list(self.stacks)
            contribution = new_pips[1-active] - new_pips[active]
            new_stacks[active] -= contribution
            new_pips[active] += contribution
            state = RoundState(self.button + 1, self.street, new_pips,
                               new_stacks, self.hands, self.deck, self.board, self, action)
            return state.proceed_street()
        if isinstance(action, CheckAction):
            if (self.street == 0 and self.button > 0) or self.button > 1 or self.street == 2 or self.street == 3:  # both players acted
                # Create intermediate state with CheckAction before proceeding to street
                temp_state = RoundState(self.button + 1, self.street, self.pips,
                                        self.stacks, self.hands, self.deck, self.board, self, action)
                return temp_state.proceed_street()
            # let opponent act
            return RoundState(self.button + 1, self.street, self.pips, self.stacks, self.hands, self.deck, self.board, self, action)
        # isinstance(action, RaiseAction)
        new_pips = list(self.pips)
        new_stacks = list(self.stacks)
        contribution = action.amount - new_pips[active]
        new_stacks[active] -= contribution
        new_pips[active] += contribution
        return RoundState(self.button + 1, self.street, new_pips, new_stacks, self.hands, self.deck, self.board, self, action)
