
"""
Created on Wed Mar  8 17:34:39 2023

@author: Brandon Botzer - btb5103


Useful Sites:
    https://gymnasium.farama.org/api/env/
    https://gymnasium.farama.org/content/basic_usage/#more-information
    https://gymnasium.farama.org/environments/box2d/#
    https://gymnasium.farama.org/environments/box2d/car_racing/
    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/car_racing.py
    


"""



#%%




#bring in the new gymnasium as gym
import gymnasium as gym

#make the enviornmnet for the box2D Car Racing v2
#make the colors change each run
#render for human [non-RGB array]
env = gym.make("CarRacing-v2", domain_randomize = True, render_mode="human")

env.reset(options={"randomize":True})

env.render()






#%%
env.close()