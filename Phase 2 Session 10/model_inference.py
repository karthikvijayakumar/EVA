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
    
    citymap = Image.open("images/MASK1.png")
    roadmask = Image.open("images/MASK1.png").convert('L')
    car_image = Image.open("images/car_upright.png")
    car_image_width, car_image_height = car_image.getbbox()[2:4]
    car_image_resized = car_image.resize( (int(car_image_width/2), int(car_image_height/2)) )
    
    work_dir = mkdir('exp', 'brs')
    monitor_dir = mkdir(work_dir, 'monitor')
    monitor_car_pov_dir = mkdir(work_dir, 'monitor_car_pov')

    file_name = 'TD3_CityMap_0'
    state_dim = env.observation_window_size
    action_dim = 1
    max_action = env.max_action

    # Selecting the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    # Loading the model
    policy = TD3(state_dim, action_dim, max_action, device)
    policy.load(file_name, './pytorch_models/')

    # Recording in map view
    env = CityMap(citymap, roadmask, car_image_resized)
    env = wrappers.Monitor(env, monitor_dir, force = True)
    env.reset()
    
    avg_reward = evaluate_policy(policy, env, eval_episodes=3)

    # Wrapup recording
    env.close()
    env.stats_recorder.save_complete()
    env.stats_recorder.done = True


    # Recording in car view
    env_car_pov = CityMap(citymap, roadmask, car_image_resized, render_pov = 'car')
    env_car_pov = wrappers.Monitor(env_car_pov, monitor_dir, force = True)
    env_car_pov.reset()
    
    avg_reward_car_pov = evaluate_policy(policy, env_car_pov, eval_episodes=3)

    # Wrapup recording
    env_car_pov.close()
    env_car_pov.stats_recorder.save_complete()
    env_car_pov.stats_recorder.done = True