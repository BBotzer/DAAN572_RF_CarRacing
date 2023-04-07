# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:38:19 2023

@author: btb51
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

#%%

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")


#%%
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    terminate = False
    score = 0
    
    while not terminate:
        env.render()
        action = env.action_space.sample()
        n_state, reward, terminate, truncate, info = env.step(action)
        score = score + reward
        
    print("Episode:{} Score:{}".format(episode, score))
    
env.close()


#%%
log_path = os.path.join('Training', 'Logs')

env = make_vec_env(env_name, n_envs=4)



model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps = 40000)





#%%
ppo_path = os.path.join('Training', 'Saved Models', 'PPO_Driving_model')

model.save(ppo_path)


#%%
evaluate_policy(model, env, n_eval_episodes=10, render=True)


#%%
env.close()





