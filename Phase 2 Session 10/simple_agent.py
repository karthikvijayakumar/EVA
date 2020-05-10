# 
#   README
# 
#   The purpose of this script is to be able to operate on the custom gym environment 
#   with manually defined simple agents/policies
#   There are 2 agents/policies implemented here:
#   1. Point to goal ( function simple_policy )
#   2. Random agent
#   There is a section below that generate the car point of view of the car to better 
#   understand what the agent sees at each step

from custom_gym_env import CityMap
import gym
from gym import error, spaces, utils, wrappers
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
from gym.envs.classic_control import rendering
from PIL import Image, ImageOps, ImageDraw
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import random
from tqdm import trange

from models import TD3
from model_helpers import evaluate_policy

'''
    This simple policy turns the car towards the goal

    Input:
        orientation: 
            Angle in radians from the axis of the car towards the goal divided by np.pi
            Ranges from -1 to 1
'''
def simple_policy(orientation):
    return np.array( [ np.float32(orientation) ] )

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path            
            
if __name__ == "__main__":
    register(
        id='citymap-v0',
        entry_point= CityMap
    )
    
    citymap = Image.open("images/citymap.png")
    roadmask = Image.open("images/MASK1.png").convert('1').convert('L')
    car_image = Image.open("images/car_upright.png")
    car_image_width, car_image_height = car_image.getbbox()[2:4]
    car_image_resized = car_image.resize( (int(car_image_width/4), int(car_image_height/4)) )
    
    work_dir = mkdir('exp', 'brs')
    monitor_dir = mkdir(work_dir, 'monitor')
    # monitor_car_pov_dir = mkdir(work_dir, 'monitor_car_pov')

    # Recording in map view
    env = CityMap(citymap, roadmask, car_image_resized, render_pov = 'map')
    env = wrappers.Monitor(env, monitor_dir, force = True, video_callable=lambda episode_id: True, write_upon_reset=False)
    state = env.reset()
 
    file_name = 'TD3_CityMap_0'
    action_dim = env.action_space.shape[0]
    max_action = env.max_action

    # Selecting the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
    done = False
    episode_reward = 0
    episode_timesteps = 0
    actions =[]
    while not done:
        screen, orientation, dist_goal = state        
        action = simple_policy(orientation)
        # action = env.action_space.sample()
        
        actions.append(action)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        episode_timesteps += 1        
        state = next_state
    print ("---------------------------------------")
    print ("Rewards per episode: ", episode_reward )
    print ("Episode length: ", episode_timesteps)
    print ("Average action: %f" % (np.mean(actions)) )
    print ("Std deviation action: %f" % (np.std(actions)) )
    print ("---------------------------------------")

    # Wrapup recording
    env.close()
    env.stats_recorder.save_complete()
    env.stats_recorder.done = True
    print(info)

    # # Recording in car view
    # env_car_pov = CityMap(citymap, roadmask, car_image_resized, render_pov = 'car')
    # env_car_pov = wrappers.Monitor(env_car_pov, monitor_car_pov_dir, force = True, video_callable=lambda episode_id: True )
    # env_car_pov.reset()
    
    #     done_bool = False
    # episode_reward = 0
    # episode_timesteps = 0
    # actions =[]
    # while not done_bool:
    #     screen, orientation, dist_goal = state
    #     action = simple_policy(orientation)
    #     actions.append(action)
    #     next_state, reward, done, info = env.step(action)

    #     episode_reward += reward
    #     done_bool = bool(done)        
    #     state = next_state
    # print ("---------------------------------------")
    # print ("Rewards per episode: ", episode_reward )
    # print ("Episode length: ", episode_timesteps)
    # print ("Average action: %f" % (np.mean(actions)) )
    # print ("Std deviation action: %f" % (np.std(actions)) )
    # print ("---------------------------------------")    

    # # Wrapup recording
    # env_car_pov.close()
    # env_car_pov.stats_recorder.save_complete()
    # env_car_pov.stats_recorder.done = True