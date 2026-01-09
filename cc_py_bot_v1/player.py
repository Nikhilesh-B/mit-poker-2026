'''
Ceylan's Pokerbot, v1
'''

from skeleton.actions import FoldAction, CallAction, CheckAction, \
RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, \
SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

from ccbot import MyBot

# GameState: bankroll, game_clock, round_num
# TerminalState: deltas
# RoundState: legal_actions(), button, street, pips, stacks, hands, board

class Player(Bot):
    ''' A pokerbot. '''

    def __init__(self, *args):
        '''
        Called when a new game starts. Called exactly once.
        '''
        # TODO: Would be adding our RL bot here.
        
        
        self.my_bankroll_list = [0]
        self.round_num = 0
        
        self.MyBot = MyBot(*args)
        

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.
        '''
        
        # Total bankroll at this point.
        self.my_bankroll = game_state.bankroll
        self.my_bankroll_list.append(self.my_bankroll)
        
        # # Seconds left for the bot to finish the game.
        # self.game_clock = game_state.game_clock
        
        # The round number from 1 to NUM_ROUNDS.
        self.round_num = game_state.round_num
        
        # BB: 1, SB: 0.
        self.button = active
        
        # Big blind discards first.
        # BB S-2, SB S-3. Street is number of cards on the board.
        
        print(f'\n---- {game_state.round_num} ----')
        print(f'Starting Bankroll: {self.my_bankroll}')
        

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.
        '''
        print('Round Completed.')
        pass
    

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        
        action = self.MyBot.get_action(game_state, round_state, active)
        print(action)
        
        return action



if __name__ == '__main__':
    run_bot(Player(), parse_args())
