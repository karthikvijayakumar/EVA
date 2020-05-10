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
    car_image_resized = car_image.resize( (int(car_image_width/2), int(car_image_height/2)) )
    
    work_dir = mkdir('exp', 'brs')
    monitor_dir = mkdir(work_dir, 'monitor')
    monitor_car_pov_dir = mkdir(work_dir, 'monitor_car_pov')

    # Recording in map view
    env = CityMap(citymap, roadmask, car_image_resized)
    env = wrappers.Monitor(env, monitor_dir, force = True, video_callable=lambda episode_id: True)
    env.reset()
 
    file_name = 'TD3_CityMap_0'
    action_dim = env.action_space.shape[0]
    max_action = env.max_action

    # Selecting the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    #Model params    
    seed = 42
    env_name = "CityMap"
    max_action = env.max_action
    action_dim = env.action_space.shape[0]
    seed = 0 # Random seed number    
    expl_noise = 0.02 # Exploration noise - STD value of exploration Gaussian noise
    batch_size = 256 # Size of the batch
    discount = 0.90 # Discount factor gamma, used in the calculation of the total discounted reward
    polyak = 0.5 # Target network update rate
    policy_noise = 0.02 # STD of Gaussian noise added to the actions for the exploration purposes during model training
    noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
    policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
    actor_lr = 0.0001
    critic_lr = 0.0001    

    # Loading the model
    policy = TD3(
        action_dim, 
        max_action, 
        batch_size = batch_size, 
        discount = discount, 
        polyak = polyak, 
        policy_noise = policy_noise, 
        noise_clip = noise_clip, 
        policy_freq = policy_freq, 
        actor_lr = actor_lr,
        critic_lr = critic_lr,
        device = device)
    policy.load(file_name, './pytorch_models/')
   
    avg_reward = evaluate_policy(policy, env, eval_episodes=3)

    # Wrapup recording
    env.close()
    env.stats_recorder.save_complete()
    env.stats_recorder.done = True


    # # Recording in car view
    env_car_pov = CityMap(citymap, roadmask, car_image_resized, render_pov = 'car')
    env_car_pov = wrappers.Monitor(env_car_pov, monitor_car_pov_dir, force = True, video_callable=lambda episode_id: True )
    env_car_pov.reset()
    
    avg_reward_car_pov = evaluate_policy(policy, env_car_pov, eval_episodes=3)

    # # Wrapup recording
    env_car_pov.close()
    env_car_pov.stats_recorder.save_complete()
    env_car_pov.stats_recorder.done = True