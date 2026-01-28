'''
Ceylan's Pokerbot, RL-v1
'''

from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

from ccbot_rl import MyRLBot

class Player(Bot):
    def __init__(self, *args):
        self.MyBot = MyRLBot(*args)
        
    def handle_new_round(self, game_state, round_state, active):
        pass
        
    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        action = self.MyBot.get_action(game_state, round_state, active)
        
        return action


if __name__ == '__main__':
    run_bot(Player(), parse_args())
