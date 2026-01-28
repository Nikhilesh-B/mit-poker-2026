''' Pokerbots RL Agent '''

# %% Import Libraries

import numpy as np

import gymnasium as gym
import poker_gym as PokerEnv

import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# %% Windows needs this __main__ thing for Vec Parallelization apparently.

if __name__ == '__main__':
    # tensorboard --logdir "C:\Users\DELL\Desktop\MIT MFin\3_IAP_2026\6.9630\mit-poker-2026\cc_py_bot_v3_rl\ppobot_v0\tboard"
    
    # %% Variables
    
    TRAIN_STEPS = int(2e6)
    CPOINT_STEPS = int(2.5e4)
    EVAL_STEPS = int(1e4)
    
    VEC_ENV = True
    N_VEC = 4
    
    # %% Env Setup
    
    if VEC_ENV:
        # n_envs=4 uses 4 CPU cores. Change this based on your machine.
        # We use SubprocVecEnv to bypass the Global Interpreter Lock (GIL) for speed.
        # Create the training envs
        env = make_vec_env(
            PokerEnv.TossHold, 
            n_envs=N_VEC, 
            vec_env_cls=SubprocVecEnv)
        
        # Create a SINGLE separate evaluation env (Wrap in Monitor for stats)
        # We don't need parallel envs for eval, just one accurate one.
        eval_env = Monitor(PokerEnv.TossHold())
        
        n_steps_x = int(4096/N_VEC)
        
    else:
        env = PokerEnv.TossHold()
        eval_env = PokerEnv.TossHold()
        
        n_steps_x = 4096
    
    # %% Custom Policy
    
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[512, 256], vf=[512, 256]))
    
    
    # %% Get Model
    
    model = PPO(
        policy='MultiInputPolicy',
        env=env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppobot_v0/tboard/",
        learning_rate=2e-4,
        n_steps=n_steps_x,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03)
    
    
    # %% Callbacks
    
    checkpoint_callback = CheckpointCallback(
        save_freq=CPOINT_STEPS,
        save_path="./ppobot_v0/checkpoints/",
        name_prefix="ppo_poker")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./ppobot_v0/best_model/",
        log_path="./ppobot_v0/eval_logs/",
        eval_freq=EVAL_STEPS,
        deterministic=True,
        render=False)
    
    # %% Training
    
    print("Starting training...")
    model.learn(
        total_timesteps=TRAIN_STEPS, 
        progress_bar=True, 
        callback=[checkpoint_callback, eval_callback])
    
    print("Saving final model...")
    model.save("./ppobot_v0/ppobot_final")
    
    env.close()
    eval_env.close()
