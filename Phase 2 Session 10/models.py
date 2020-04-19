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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3,padding = 1, stride =2)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4,8, kernel_size=3, padding = 1, stride =2)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8,16, kernel_size=3, padding = 1, stride =2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(32)
        self.head = nn.Linear(32, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.avg_pool2d( x, 3 )
        return self.max_action * torch.tanh(self.head(x.view( x.size(0),-1)))
    
class Critic(nn.Module):

  def __init__(self, state_dim, action_dim, max_action):
    super(Critic, self).__init__()
    self.max_action = max_action
        
    # Defining the first Critic neural network
    self.critic1_conv1 = nn.Conv2d(1, 4, kernel_size=3,padding = 1, stride =2)
    self.critic1_bn1 = nn.BatchNorm2d(4)
    self.critic1_conv2 = nn.Conv2d(4,8, kernel_size=3, padding = 1, stride =2)
    self.critic1_bn2 = nn.BatchNorm2d(8)
    self.critic1_conv3 = nn.Conv2d(8,16, kernel_size=3, padding = 1, stride =2)
    self.critic1_bn3 = nn.BatchNorm2d(16)
    self.critic1_conv4 = nn.Conv2d(16, 32, kernel_size=3)
    self.critic1_bn4 = nn.BatchNorm2d(32)
    self.critic1_head = nn.Linear(33, 1)
    # Critic gives out only 1 value hence the output dimension is one
    # Critic also takes action as input which is a scalar, hence adding 1 to the output of the conv layers
    
    # Defining the second Critic neural network
    self.critic2_conv1 = nn.Conv2d(1, 4, kernel_size=3,padding = 1, stride =2)
    self.critic2_bn1 = nn.BatchNorm2d(4)
    self.critic2_conv2 = nn.Conv2d(4,8, kernel_size=3, padding = 1, stride =2)
    self.critic2_bn2 = nn.BatchNorm2d(8)
    self.critic2_conv3 = nn.Conv2d(8,16, kernel_size=3, padding = 1, stride =2)
    self.critic2_bn3 = nn.BatchNorm2d(16)
    self.critic2_conv4 = nn.Conv2d(16, 32, kernel_size=3)
    self.critic2_bn4 = nn.BatchNorm2d(32)
    self.critic2_head = nn.Linear(33, 1)
    # Critic gives out only 1 value hence the output dimension is one
    # Critic also takes action as input which is a scalar, hence adding 1 to the output of the conv layers

  def forward(self, x, u):
    ###############
    ## Critic 1
    ###############
    # Pass through convolutional layers
    x1 = F.relu(self.critic1_bn1(self.critic1_conv1(x)))
    x1 = F.relu(self.critic1_bn2(self.critic1_conv2(x1)))
    x1 = F.relu(self.critic1_bn3(self.critic1_conv3(x1)))
    x1 = F.relu(self.critic1_bn4(self.critic1_conv4(x1)))
    x1 = F.avg_pool2d( x1, 3 ) 
    x1 = x1.view(x1.size(0), -1)
    
    #Concatenate action with the output of the convolutional layers
    x1 = torch.cat([x1, u], 1)
    
    #Pass through FC layer
    x1 = self.critic1_head(x1)

    ###############
    # Critic 2
    ###############
    
    #Pass through convolutional layers
    x2 = F.relu(self.critic2_bn1(self.critic2_conv1(x)))
    x2 = F.relu(self.critic2_bn2(self.critic2_conv2(x2)))
    x2 = F.relu(self.critic2_bn3(self.critic2_conv3(x2)))
    x2 = F.relu(self.critic2_bn4(self.critic2_conv4(x2)))
    x2 = F.avg_pool2d( x2, 3 ) 
    x2 = x2.view(x2.size(0), -1)
 
    #Concatenate action with the output of the convolutional layers
    x2 = torch.cat([x2, u], 1)
    
    #Pass through FC layer
    x2 = self.critic2_head(x2)
    
    return x1, x2

  def Q1(self, x, u):   
    ###############
    ## Critic 1
    ###############
    # Pass through convolutional layers
    x1 = F.relu(self.critic1_bn1(self.critic1_conv1(x)))
    x1 = F.relu(self.critic1_bn2(self.critic1_conv2(x1)))
    x1 = F.relu(self.critic1_bn3(self.critic1_conv3(x1)))
    x1 = F.relu(self.critic1_bn4(self.critic1_conv4(x1)))
    x1 = F.avg_pool2d( x1, 3 )
    x1 = x1.view(x1.size(0), -1)
 
    #Concatenate action with the output of the convolutional layers
    x1 = torch.cat([x1, u], 1)
    
    #Pass through FC layer
    x1 = self.critic1_head(x1)
    
    return x1


class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action, device):
    self.device = device
    self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim, max_action).to(self.device)
    self.critic_target = Critic(state_dim, action_dim, max_action).to(self.device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state):
    state = torch.Tensor(state).to(self.device)
    return self.actor(state).detach().cpu().data.numpy().flatten()[0]

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    # print( "Replay buffer size at beginning of call to train: " + str(len(replay_buffer.storage)) )
    actor_conv1_frozen = self.actor.conv1.weight.data.clone().detach()
    actor_conv2_frozen = self.actor.conv2.weight.data.clone().detach()
    actor_head_frozen = self.actor.head.weight.data.clone().detach()

    for it in trange(iterations, desc = "TD3 train function loop"):
      # print("Training iteration local: " + str(it))
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      
      # Each element obtained from the replay_buffer is a np array
      # Each element of the batch_states, batch_next_states is of the shape (1,1,size,size)
      # Batch_states and batch_next_states have shapes (batch_size,1,1,size,size)
      # What we want to feed to the conv networks is (batch_size, 1, size, size)
      # torch.Tensor(batch_states) gives us a tensor of the shape (batch_size,1,1,size,size)
      # Hence we squeeze out the first dimension ( zero'th dimension of size batch_size is intact )

      # Each element of batch_rewards, batch_actions, batch_dones is a scalar
      # We need to give a column vector to pytorch to compute losses
      # Hence scalars go through a reshape(-1,1) ( eg: [1,2,3] to [[1],[2],[3]] )

      state = torch.Tensor(batch_states).squeeze(1).to(self.device)
      next_state = torch.Tensor(batch_next_states).squeeze(1).to(self.device) 
      action = torch.Tensor(batch_actions).view(-1,1).to(self.device) #Convert row vector to column vector eg: [1,2,3] to [[1],[2],[3]]
      reward = torch.Tensor(batch_rewards).view(-1,1).to(self.device) #Convert row vector to column vector eg: [1,2,3] to [[1],[2],[3]]
      done = torch.Tensor(batch_dones).view(-1,1).to(self.device) #Convert row vector to column vector eg: [1,2,3] to [[1],[2],[3]]
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      # noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(self.device)
      # noise = noise.clamp(-noise_clip, noise_clip)
      # noise = noise.reshape(-1,1)
      noise_distribution = torch.distributions.normal.Normal(0, policy_noise)
      noise = noise_distribution.sample(torch.Size([batch_size])).clamp(-noise_clip,noise_clip).view(-1,1).to(self.device)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # print("Action shape: " + str(action.shape))
      # print("Next action shape: " + str(next_action.shape))
        
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:        

        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()        
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    #Compute change in weights after the training iterations
    actor_conv1_total_change = torch.sum( (torch.abs(self.actor.conv1.weight.data - actor_conv1_frozen) ) ).detach().cpu().numpy().flatten()[0]
    actor_conv2_total_change = torch.sum( (torch.abs(self.actor.conv2.weight.data - actor_conv2_frozen) ) ).detach().cpu().numpy().flatten()[0]
    actor_head_total_change = torch.sum( (torch.abs(self.actor.head.weight.data - actor_head_frozen) ) ).detach().cpu().numpy().flatten()[0]

    print("Actor Conv 1 total change: ", actor_conv1_total_change )
    print("Actor Conv 2 total change: ", actor_conv2_total_change )
    print("Actor head total change: ", actor_head_total_change )
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))