''' Pokerbots RL Agent '''

# %% 

import gymnasium as gym
import poker_gym as PokerEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy


env = PokerEnv.TossHold()
print(check_env(env))

model = PPO("MultiInputPolicy", env, verbose=2, 
            tensorboard_log="./ppo_pokerbot_v0/")
model.learn(total_timesteps=int(5e3), progress_bar=True)

model.save("ppo_pokerbot_v0_model_saved")

# mean_reward, std_reward = evaluate_policy(model, model.get_env(), 
#                                           n_eval_episodes=20)


# vec_env = model.get_env()
# obs = vec_env.reset()

# for i in range(15):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human") 