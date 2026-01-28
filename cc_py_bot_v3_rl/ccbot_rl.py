''' This class is submits actions in player.py. '''

from skeleton.actions import FoldAction, CallAction, CheckAction, \
RaiseAction, DiscardAction

import poker_utils_rl

from stable_baselines3 import PPO


class MyRLBot():
    def __init__(self):
        self.model = PPO.load(r"C:\Users\DELL\Desktop\MIT MFin\3_IAP_2026\6.9630\mit-poker-2026\cc_py_bot_v2_rl\ppo_pokerbot_v0\best_model_v2\best_model.zip")
        
    def handle_new_round(self):
        pass
    
    def handle_round_over(self, game_state, terminal_state, active):
        pass
    
    def get_action(self, game_state, round_state, active):
        obs = poker_utils_rl.state_to_obs(game_state, round_state, active)
        action_rlbot, _ = self.model.predict(obs, deterministic=True)
        
        action = poker_utils_rl.decode_rl_action(action_rlbot, round_state)
        
        return action
        

    