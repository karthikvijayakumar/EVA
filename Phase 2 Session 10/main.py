#Generic imports
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm, trange
import os

# Importing self written classes
from models import TD3
from model_helpers import ReplayBuffer, evaluate_policy
from custom_gym_env import CityMap

#Helper function
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Setting params
env_name = "CityMap"
seed = 0 # Random seed number
start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 3e4 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = (0.2*np.pi)/(2.7*180) # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = (0.2*np.pi)/(2.7*180) # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
train_iterations = 100 # Number of iterations to run the training cycle for each time an episide is over


#File name for actor and critic model saved files
file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

#Create folder to save trained models
if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")

#Create the environment
citymap = Image.open("images/MASK1.png")
roadmask = Image.open("images/MASK1.png").convert('L')
car_image = Image.open("images/car_upright.png")
car_image_width, car_image_height = car_image.getbbox()[2:4]
car_image_resized = car_image.resize( (int(car_image_width/2), int(car_image_height/2)) )

work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

env = CityMap(citymap, roadmask, car_image_resized)

# Set seeds and get get info to initiate classes
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_window_size
action_dim = 1
max_action = env.max_action

#Create the policy network
policy = TD3(state_dim, action_dim, max_action, device)

#Create the experience replay buffer
replay_buffer = ReplayBuffer()

#List for storing model evaluations
evaluations = [evaluate_policy(policy, env)]
evaluation_timesteps = [0]

#Initialise variables
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
episode_timesteps = 1e3
done = True
t0 = time.time()

#Training

# We start the main loop over max_timestep timesteps
while total_timesteps < max_timesteps:
  
  # If the episode is done
  if done:

    # If we are not at the very beginning, we start the training process of the model
    if total_timesteps != 0:
      print("\n\nTotal Timesteps: {} Episode Num: {} Episode length: {} Reward: {}".format(total_timesteps, episode_num, episode_timesteps, episode_reward))
      policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

    # We evaluate the episode and we save the policy
    if timesteps_since_eval >= eval_freq:
      timesteps_since_eval %= eval_freq
      evaluations.append(evaluate_policy(policy, env))
      evaluation_timesteps.append(total_timesteps)
      policy.save(file_name, directory="./pytorch_models")
      np.save("./results/%s" % (file_name), evaluations)
    
    # When the training step is done, we reset the state of the environment
    screen, orientation, dist_goal = env.reset()
    
    # Set the Done to False
    done = False
    
    # Set rewards and episode timesteps to zero
    episode_reward = 0
    episode_timesteps = 0
    episode_num += 1
  
  # Before start_timesteps timesteps, we play random actions
  if total_timesteps < start_timesteps:
    action = env.action_space.sample()[0]
  else: # After 10000 timesteps, we switch to the model
    action = policy.select_action((screen, orientation, dist_goal))
    # If the explore_noise parameter is not 0, we add noise to the action and we clip it
    if expl_noise != 0:
      action = np.clip( (action + np.random.normal(0, expl_noise)), env.action_space.low, env.action_space.high)[0]
      # np.clip returns an array. We need a scalar. Hence taking the first element
  
  # The agent performs the action in the environment, then reaches the next state and receives the reward
  (new_screen, new_orientation, new_dist_goal), reward, done, _ = env.step(action)

  # We check if the episode is done
  done_bool = 1 if episode_timesteps + 1 == env.max_episode_steps else float(done)
  done = bool(done_bool)

  # We increase the total reward
  episode_reward += reward
  
  # We store the new transition into the Experience Replay memory (ReplayBuffer)
  replay_buffer.add((screen, orientation, dist_goal, new_screen, new_orientation, new_dist_goal, action, reward, done_bool))

  # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
  screen = new_screen
  orientation = new_orientation
  dist_goal = new_dist_goal
  episode_timesteps += 1
  total_timesteps += 1
  timesteps_since_eval += 1

env.close()

# We add the last policy evaluation to our list of evaluations and we save our model
evaluations.append(evaluate_policy(policy, env))
evaluation_timesteps.append(total_timesteps)

if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
np.save("./results/%s" % (file_name), evaluations)