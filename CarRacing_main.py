
"""
Created on Wed Mar  8 17:34:39 2023

@author: Brandon Botzer - btb5103


Useful Sites:
    https://gymnasium.farama.org/api/env/
    https://gymnasium.farama.org/content/basic_usage/#more-information
    https://gymnasium.farama.org/environments/box2d/#
    https://gymnasium.farama.org/environments/box2d/car_racing/
    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/car_racing.py
    https://github.com/DLR-RM/stable-baselines3/blob/master/README.md
    


"""



#%%




#bring in the new gymnasium as gym
import gymnasium as gym

from gym.envs.box2d import CarRacing

#make the enviornmnet for the box2D Car Racing v2
#make the colors change each run
#render for human [non-RGB array]
env = gym.make("CarRacing-v2", domain_randomize = True, render_mode="rgb_array")

env.reset(options={"randomize":True})

env.render()




#%%
#Test of using the stable baselines library

import gymnasium as gym

from gym.envs.box2d import CarRacing


#from stable_baselines3.common.policies import CnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


if __name__ == '__main__':

    env2 = lambda : CarRacing(
        grayscale=1,
        show_info_panel=0,
        discretize_actions="hard",
        frames_per_state=4,
        num_lanes=1,
        num_tracks=1)

    #env = getattr(environments, env)
    env2 = make_vec_env("CarRacing-v0", n_envs=4)


    model = PPO("CnnPolicy", env2, verbose=1)
    #model = PPO.load('CarRacing-v0')

    model.set_env(env2)

    obs = env2.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env2.step(action)
        env2.render()








#%%
env.close()