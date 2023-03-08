# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:34:39 2023

@author: btb51
"""

#bring in the new gymnasium as gym
import gymnasium as gym

#make the enviornmnet for the box2D Car Racing v2
#make the colors change each run
#render for human [non-RGB array]
env = gym.make("CarRacing-v2", domain_randomize = True, render_mode="human")

env.reset(options={"randomize":True})

env.render()

env.close()