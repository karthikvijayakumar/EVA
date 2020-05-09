import os
import numpy as np
import random

#Helper function
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    sample_tuples = random.sample(self.storage, batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = tuple(zip(*sample_tuples))

    return batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones

def evaluate_policy(policy, env, eval_episodes = 3):  
  episode_rewards = []
  episode_lengths = []  
  actions = []
  for episode_num in range(eval_episodes):
    obs = env.reset()
    done = False
    curr_episode_reward = 0
    curr_episode_length = 0
    while not done:
      action = policy.select_action(obs)
      actions.append(action)
      obs, reward, done, info = env.step(action)
      curr_episode_reward += reward
      curr_episode_length += 1
    
    episode_rewards.append(curr_episode_reward)
    episode_lengths.append(curr_episode_length)
  avg_reward = np.mean(episode_rewards)
  print ("---------------------------------------")
  print ("Episode lengths: ", episode_lengths )
  print ("Rewards per episode: ", episode_rewards )
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("Average action: %f" % (np.mean(actions)) )
  print ("Std deviation action: %f" % (np.std(actions)) )
  print ("---------------------------------------")
  return avg_reward