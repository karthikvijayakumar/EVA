#Generic imports
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from PIL import Image, ImageOps
import pickle
from tqdm import trange
import seaborn as sns

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

# Setup gym environment
citymap = Image.open("images/citymap.png")
roadmask = Image.open("images/MASK1.png").convert('1').convert('L')
# Converting to binary and back to grayscale to ensure there are only full black and white pixels
car_image = Image.open("images/car_upright.png")
car_image_width, car_image_height = car_image.getbbox()[2:4]
car_image_resized = car_image.resize( (int(car_image_width/2), int(car_image_height/2)) )

env = CityMap(citymap, roadmask, car_image_resized)

###########
# Setup model attributes
############

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Setting params
seed = 42
env_name = "CityMap"
max_action = env.max_action
action_dim = env.action_space.shape[0]
seed = 0 # Random seed number
start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e4 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.02 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 256 # Size of the batch
discount = 0.90 # Discount factor gamma, used in the calculation of the total discounted reward
polyak = 0.5 # Target network update rate
policy_noise = 0.02 # STD of Gaussian noise added to the actions for the exploration purposes during model training
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
actor_lr = 0.0001
critic_lr = 0.0001
train_iterations = 100 # Number of iterations to run the training cycle for each time an episide is over
# Params for epsilon greedy random action
eps_start = 1.0
eps_end = 0.05
eps_decay = 5000

#Creating policy object
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
    device = device )

#File name for saving actor and critic models
file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")


#Create directories for storing models and evaluations
mkdir('.', 'pytorch_models')
mkdir('.', 'results')

#Set random seeds
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

#Create the experience replay buffer
replay_buffer = ReplayBuffer(max_size=int(max_timesteps/2))


env.reset() #Reset env before evaluating
#List for storing model evaluations
evaluations = [evaluate_policy(policy, env)]
evaluation_timesteps = [0]
#Reset env after evaluation
env.reset()

# Interaction with environment and subsequent training
episode_num = 0
episode_timesteps = 0
episode_reward = 0
done = True
info = {}
timesteps_since_eval = 0


for timestep in trange(int(max_timesteps)):

    assert episode_timesteps == env.num_steps, "Env.num_steps and episode_timesteps are out of sync; env.num_steps" + str(env.num_steps) + "; episode_timesteps : " + str(episode_timesteps)
    assert episode_timesteps <= env.max_episode_steps, "Episode exceeding max length " + str(env.max_episode_steps) + "; Episode timesteps = " + str(episode_timesteps) 

    if(done):        
        # Update the policy after an episode has completed
        if(timestep>0):
            print(
                "Total Timesteps: {} Episode Num: {} Episode length: {} Reward: {} Info: {}".format(
                timestep, episode_num, env.num_steps, np.round(episode_reward,2), info)
            )
            if(len(replay_buffer.storage) > batch_size):
                policy.update_policy(replay_buffer, episode_timesteps)
            else:
                print("Replay buffer length too small : " + str(len(replay_buffer.storage)) )

        #Save model
        policy.save(file_name, directory="./pytorch_models")

        #Evaluate policy if its time
        if(timesteps_since_eval >= eval_freq):
            print("Evaluating policy")
            env.reset() #Reset env before evaluation
            timesteps_since_eval = timesteps_since_eval%eval_freq
            evaluations.append(evaluate_policy(policy, env))
            evaluation_timesteps.append(timestep)
            np.save("./results/%s" % (file_name), evaluations)
            env.reset() #Reset env post evaluation

        state = env.reset()
        episode_num += 1
        episode_reward =0
        episode_timesteps = 0
        done = False        
    
    if(done):
        assert episode_timesteps == 0, "Episode is done but episode_timesteps not reset"

    if timestep < start_timesteps:
        action = env.action_space.sample()
        #Taking first element since we need a scalar
    else:
        # Compute epsilon for epsilon greedy random action
        eps = eps_end + ( (eps_start-eps_end)*np.exp( (start_timesteps-timestep)/eps_decay )  )        
        if( np.random.uniform(0,1) < eps ):
            action = env.action_space.sample()        
        else:
            action = policy.select_action(state)
        # action returned is an np array of size 1 with dtype no float32
        
        # Add noise to action
        # If the explore_noise parameter is not 0, we add noise to the action and we clip it
        if expl_noise != 0:
            action = np.clip( (action + np.random.normal(0, expl_noise)), env.action_space.low, env.action_space.high )
        
    #Perform action on environment
    next_state, reward, done, info = env.step(action)
    
    if(not(done) and reward > 1):
        raise Exception("High reward for non-terminal state: Reward = " + str(reward) + '; done = ' + str(done) + '; dist_from_goal = ' + str(env.distance_from_goal))

    if(done):
        print("Episode done")
    done_float = np.float32(done)
    replay_buffer.add( (state, next_state, action, reward, done_float) )

    state = next_state
    episode_timesteps += 1
    episode_reward += reward
    timesteps_since_eval += 1
    # End of for loop

env.close()

# We add the last policy evaluation to our list of evaluations and we save our model
evaluations.append(evaluate_policy(policy, env))
evaluation_timesteps.append(max_timesteps)

if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
np.save("./results/%s" % (file_name), evaluations)

fig, ax = plt.subplots(1,1, figsize = (12,9))
sns.lineplot(evaluation_timesteps, evaluations, ax = ax)
plt.savefig('training_eval.png', bbox_inches = 'tight')

# Model inference
monitor_dir = mkdir('.', 'monitor')
monitor_car_pov_dir = mkdir('.', 'monitor_car_pov')

# Env setup
inference_env_map = CityMap(citymap, roadmask, car_image_resized)
inference_env_map = wrappers.Monitor(inference_env_map, monitor_dir, force = True, video_callable=lambda episode_id: True)
inference_env_map.reset()

inference_env_car = CityMap(citymap, roadmask, car_image_resized, render_pov = 'car')
inference_env_car = wrappers.Monitor(inference_env_car, monitor_car_pov_dir, force = True, video_callable=lambda episode_id: True)
inference_env_car.reset()

#Loading models
# Loading the model
inference_policy = TD3(
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
inference_policy.load(file_name, './pytorch_models/')

#Map POV
avg_reward_map = evaluate_policy(inference_policy, inference_env_map, eval_episodes=3)

# Wrapup recording
inference_env_map.close()
inference_env_map.stats_recorder.save_complete()
inference_env_map.stats_recorder.done = True

# #Car POV
avg_reward_car = evaluate_policy(inference_policy, inference_env_car, eval_episodes=3)

# Wrapup recording
inference_env_car.close()
inference_env_car.stats_recorder.save_complete()
inference_env_car.stats_recorder.done = True
