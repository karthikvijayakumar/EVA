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
import random

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
    return np.array(batch_states), batch_next_states, batch_actions, batch_rewards, batch_dones
    
    # ind = np.random.randint(0, len(self.storage), size=batch_size)
    # batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    # for i in ind: 
    #   state, next_state, action, reward, done = self.storage[i]
    #   batch_states.append(np.array(state, copy=False))
    #   batch_next_states.append(np.array(next_state, copy=False))
    #   batch_actions.append(np.array(action, copy=False))
    #   batch_rewards.append(np.array(reward, copy=False))
    #   batch_dones.append(np.array(done, copy=False))
    # return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards), np.array(batch_dones)


def evaluate_policy(policy, env, eval_episodes=3):
  avg_reward = 0.
  for _ in range(eval_episodes):
    obs = env.reset()
    # print("Observatin shape from evaluate policy")
    # print(type(obs))
    # print(obs.shape)
    done = False
    # print("Entering while loop")
    while not done:
      action = policy.select_action(obs)
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward