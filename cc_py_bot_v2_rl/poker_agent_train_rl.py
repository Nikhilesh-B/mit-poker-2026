''' Pokerbots RL Agent '''

# %% Import Libraries

import gymnasium as gym
import poker_gym as PokerEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# %% Load Environment

print("Creating environment...")
env = PokerEnv.TossHold()
# check_env(env)

# %% Manual Test

obs, info = env.reset()
print("Initial observation shape:", {k: v.shape for k, v in obs.items()})

for i in range(10):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: reward={reward:.4f}, terminated={terminated}")
    if terminated:
        obs, info = env.reset()
        break

print("Manual test passed!")

# %% Train Model

# Tensorboard CMD: 
    # tensorboard --logdir "C:\Users\DELL\Desktop\MIT MFin\3_IAP_2026\6.9630\mit-poker-2026\cc_py_bot_v2_rl\ppo_pokerbot_v0\tboard"

print("Creating model...")
model = PPO("MultiInputPolicy", env, verbose=1, 
            tensorboard_log="./ppo_pokerbot_v0/tboard/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./ppo_pokerbot_v0/checkpoints/",
    name_prefix="ppo_poker")

eval_env = PokerEnv.TossHold()
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./ppo_pokerbot_v0/best_model/",
    log_path="./ppo_pokerbot_v0/eval_logs/",
    eval_freq=5000,
    deterministic=True,
    render=False)

print("Starting training...")
model.learn(total_timesteps=int(1e6), progress_bar=True, 
            callback=[checkpoint_callback, eval_callback])

print("Saving model...")
model.save("./ppo_pokerbot_v0/ppo_pokerbot_v0_model_saved")

# %% Evaluate

print("Evaluating trained model...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()
eval_env.close()

print("Training complete!")
