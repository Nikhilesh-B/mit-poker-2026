''' Pokerbots RL Agent '''

# %% 

import gymnasium as gym
import poker_gym as PokerEnv

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


env = PokerEnv.TossHold()

model = PPO("MultiInputPolicy", env, verbose=2)
model.learn(total_timesteps=int(5e4), progress_bar=True)

# mean_reward, std_reward = evaluate_policy(model, model.get_env(), 
#                                           n_eval_episodes=20)


# vec_env = model.get_env()
# obs = vec_env.reset()

# for i in range(15):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human") 