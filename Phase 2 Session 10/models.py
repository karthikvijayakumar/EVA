# 
# README
# 
# The TD3 implementation below heavily derives from OpenAI's spinning up [ https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/td3 ]
# As of now I have decided to entirely do away with all convolutional layers
# In Atari or other envs the image passed to the agent is not binary in nature
# And requires feature extraction and simplification to a point it can be used for classification etc.
# In our case the roadmask is already binary in nature and doesnt really require further processing
# A simple average pooling of the screen grab returned by the agent tells us in 
# which sections the road is and where it is not. This information is enough for the agent to learn.
# Potentially with some convolutions it can learn to identify horizontal and vertical lines
# Given the agent learns without them I dont see the need as of now

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from tqdm import trange

class Actor(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()        
        self.avg_pool = nn.AvgPool2d(kernel_size = 4, stride=4)
        self.linear_1 = nn.Linear(25+2,400)
        self.linear_2 = nn.Linear(400,300)        
        self.head = nn.Linear(300, action_dim)
        # 25 values from the avg pooling of a 20x20 matrix with a 4x4 kernel 
        self.max_action = max_action

    def forward(self, state):
        screen, orientation, dist_goal = state
        x = self.avg_pool( screen )
        x = x.view(x.size(0), -1)

        #Concatenate orientation and distance to goal with the output of the convolutional layers
        x = torch.cat([x, orientation, dist_goal], 1)

        # Run it through FC layer        
        x = F.relu(self.linear_1(x))                
        x = F.relu(self.linear_2(x))
        x = self.head(x)        
        x = torch.tanh(x)        

        return self.max_action * x

class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.critic1_avg_pool = nn.AvgPool2d(kernel_size = 4, stride=4)

        self.critic1_linear_1 = nn.Linear(25+2+1, 400)
        self.critic1_linear_2 = nn.Linear(400, 300)         
        self.critic1_head = nn.Linear(300, 1)
        # 25 values from the avg pooling of a 20x20 matrix with a 4x4 kernel 
        # 2 additional inputs - orientation and distance from goal, 
        # 1 input - action
        # Critic gives out only 1 value hence the output dimension is one

        # Defining the second Critic neural network
        self.critic2_avg_pool = nn.AvgPool2d(kernel_size = 4, stride=4)
        self.critic2_linear_1 = nn.Linear(25+2+1, 400)
        self.critic2_linear_2 = nn.Linear(400, 300) 
        self.critic2_head = nn.Linear(300, 1)
        # 25 values from the avg pooling of a 20x20 matrix with a 4x4 kernel 
        # 2 additional inputs - orientation and distance from goal, 
        # 1 input - action
        # Critic gives out only 1 value hence the output dimension is one

    def forward(self, state, action):
        screen, orientation, dist_goal = state
        ###############
        ## Critic 1
        ###############
        # Pass through convolutional layers
        x1 = self.critic1_avg_pool( screen ) 
        x1 = x1.view(x1.size(0), -1)

        #Concatenate action with the output of the convolutional layers
        x1 = torch.cat([x1, orientation, dist_goal, action], 1)

        #Pass through FC layer
        x1 = F.relu( self.critic1_linear_1(x1) )
        x1 = F.relu( self.critic1_linear_2(x1) ) 
        x1 = self.critic1_head(x1)

        ###############
        # Critic 2
        ###############

        #Pass through convolutional layers
        x2 = self.critic2_avg_pool( screen ) 
        x2 = x2.view(x2.size(0), -1)

        #Concatenate action with the output of the convolutional layers
        x2 = torch.cat([x2, orientation, dist_goal, action], 1)

        #Pass through FC layer
        x2 = F.relu( self.critic2_linear_1(x2) )
        x2 = F.relu( self.critic2_linear_2(x2) )
        x2 = self.critic2_head(x2)

        return x1, x2

    def Q1(self, state, action):   
        screen, orientation, dist_goal = state
        ###############
        ## Critic 1
        ###############
        # Pass through convolutional layers
        x1 = self.critic1_avg_pool( screen )
        x1 = x1.view(x1.size(0), -1)

        #Concatenate action with the output of the convolutional layers
        x1 = torch.cat([x1, orientation, dist_goal, action], 1)

        #Pass through FC layer
        x1 = F.relu( self.critic1_linear_1(x1) )
        x1 = F.relu( self.critic1_linear_2(x1) )
        x1 = self.critic1_head(x1)

        return x1
    
class TD3(object):

    def __init__(self, action_dim, max_action, batch_size, discount, polyak, policy_noise, noise_clip, policy_freq, actor_lr, critic_lr, device):
        self.device = device
        self.actor = Actor(action_dim, max_action).float().to(self.device)
        self.actor_target = Actor(action_dim, max_action).float().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size = 2500, gamma = 0.5)
        self.critic = Critic().float().to(self.device)
        self.critic_target = Critic().float().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size = 5000, gamma = 0.5)
        self.batch_size = batch_size
        self.discount = discount
        self.polyak = polyak
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_action = max_action
        self.action_dim = action_dim

        #Turn off gradients for target models to ensure they only update via polyak averaging
        for param in self.critic_target.parameters():
            param.requires_grad = False
        for param in self.actor_target.parameters():
            param.requires_grad = False

    """
    Input params:
    state - Tuple with 3 components ( screen, action, orientation )
    """
    def select_action(self, state):
        screen, orientation, dist_goal = state
        # orientation and dist_goals are scalars
        screen = torch.FloatTensor(screen).to(self.device)
        orientation = torch.FloatTensor([orientation]).view(-1,1).to(self.device) 
        dist_goal = torch.FloatTensor([dist_goal]).view(-1,1).to(self.device)
        with torch.no_grad():
            # torch.nograd should technically not be required since its a brand new constructed tensor
            # However playing safe here
            result = self.actor( (screen,orientation,dist_goal) ).cpu().detach().numpy().squeeze(0)
            # result = self.actor( (screen,orientation,dist_goal) ).cpu().data.numpy().flatten()[0]
        return result

    def compute_critic_loss(self, state, next_state, action, reward, done):
        #Current Q values( critic )
        current_q1, current_q2 = self.critic(state, action)    

        # Computing target Q values from bellman equation
        with torch.no_grad():
            target_next_action = self.actor_target(next_state)

        #Add noise to next action from target actor
        noise_distribution = torch.distributions.normal.Normal(0, self.policy_noise)
        noise = noise_distribution.sample(torch.Size([self.batch_size])).clamp(-self.noise_clip,self.noise_clip).view(-1,1).to(self.device)
        target_next_action = (target_next_action + noise).clamp(-self.max_action, self.max_action)

        #Min Q values from target critic
        target_q1, target_q2 = self.critic_target(next_state, target_next_action)
        min_target_q = torch.min(target_q1, target_q2)

        #Target Q value
        target_q = reward + (1-done)*self.discount*min_target_q

        #Loss
        loss_critic = F.smooth_l1_loss(current_q1, target_q) + F.smooth_l1_loss(current_q2, target_q)
        loss_info = dict( 
            q1_vals = current_q1.cpu().detach().numpy(),
            q2_vals = current_q2.cpu().detach().numpy(), 
        )

        return loss_critic, loss_info

    # Actor loss for a state is the critic value for that state
    def compute_actor_loss(self, state):
        return -self.critic.Q1(state, self.actor(state)).mean()

    def update_critic(self, data, iteration):

        state, next_state, action, reward, done = data

        screen, orientation, dist_goal = state
        next_screen, next_orientation, next_dist_goal = state    

        # Compute critic loss and update critic
        self.critic_optimizer.zero_grad()
        critic_loss, critic_loss_info = self.compute_critic_loss(state, next_state, action, reward, done)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_lr_scheduler.step()

    def update_actor(self, state ):
        #Freeze critics
        for param in self.critic.parameters():
            param.requires_grad = False

        self.actor_optimizer.zero_grad()
        actor_loss = self.compute_actor_loss(state)
        # print(actor_loss)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_scheduler.step()

        #Unfreeze critics
        for param in self.critic.parameters():
            param.requires_grad = True

    def update_target_networks(self):

        with torch.no_grad():
            #Update critic
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters() ):
                target_param.data.mul_(self.polyak)
                target_param.data.add_( (1-self.polyak)*param.data )

            #Update actor
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters() ):
                target_param.data.mul_(self.polyak)
                target_param.data.add_( (1-self.polyak)*param.data )


    def update_policy(self, replay_buffer, iterations):

        # Save state of actor and critic at the start of the training cycle        
        critic_frozen = Critic().float().to(self.device)
        critic_frozen.load_state_dict(self.critic.state_dict())

        actor_frozen = Actor(self.action_dim, self.max_action).float().to(self.device)       
        actor_frozen.load_state_dict( self.actor.state_dict() )

        for it in trange(iterations, desc = "T3D train function loop" ):

            state, next_state, action, reward, done = replay_buffer.sample(self.batch_size)

            screen, orientation, dist_goal = tuple(zip(*state))
            next_screen, next_orientation, next_dist_goal = tuple(zip(*next_state))

            #Converting inputs from replay buffer into pytorch tensors
            screen = torch.FloatTensor(screen).squeeze(1).to(self.device)
            orientation = torch.FloatTensor(orientation).view(-1,1).to(self.device)
            dist_goal = torch.FloatTensor(dist_goal).view(-1,1).to(self.device)

            next_screen = torch.FloatTensor(next_screen).squeeze(1).to(self.device)
            next_orientation = torch.FloatTensor(next_orientation).view(-1,1).to(self.device)
            next_dist_goal = torch.FloatTensor(next_dist_goal).view(-1,1).to(self.device)

            reward = torch.FloatTensor(reward).view(-1,1).to(self.device)
            action = torch.FloatTensor(action).view(-1,1).to(self.device)
            done = torch.FloatTensor(done).view(-1,1).to(self.device)

            state = (screen, orientation, dist_goal)
            next_state = (next_screen, next_orientation, next_dist_goal)

            #Compute critic loss and update critic
            self.update_critic((state, next_state, action, reward, done), it)

            if( it % self.policy_freq == 0):
                self.update_actor(state)
                self.update_target_networks()
        
        # Check how much the model has changed at the end of the training cycle
        critic_relevant_params = list( filter( lambda x: ('weight' in x) or ('bias' in x) , self.critic.state_dict().keys() ) )
        actor_relevant_params = list( filter( lambda x: ('weight' in x) or ('bias' in x) , self.actor.state_dict().keys() ) )

        critic_change = []
        actor_change = []
        
        for param in critic_relevant_params:
            critic_change.append( np.abs(self.critic.state_dict()[param].cpu().detach() - critic_frozen.state_dict()[param].cpu().detach()))

        for param in actor_relevant_params:
            actor_change.append( np.abs( self.actor.state_dict()[param].cpu().detach() - actor_frozen.state_dict()[param].cpu().detach() ) )

        # critic_change and actor_change are now a list of tensors
        # There will be one element in the list for each layer in the networks

        print("Sum absolute change critic: ", np.sum( [ torch.sum( torch.flatten( torch.abs(x) ) ) for x in critic_change ] ) )
        print("Sum absolute change actor: ", np.sum( [ torch.sum( torch.flatten( torch.abs(x) ) ) for x in actor_change ] ) )
        del(critic_change)
        del(actor_change)
        del(actor_frozen)
        del(critic_frozen)

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))