import torch
import torch.nn as nn
import numpy as np
from model import Model
from util import *



class CommNet(Model):

    def __init__(self, args):
        super(CommNet, self).__init__(args)

    def construct_model(self):
        '''
        define the model of vanilla CommNet
        '''
        # encoder transforms observation to latent variables
        self.encoder = nn.Linear(self.args.obs_size, self.args.hid_size)
        # communication mask where the diagnal should be 0
        self.comm_mask = cuda_wrapper(torch.ones(self.args.agent_num, self.args.agent_num) - torch.eye(self.args.agent_num, self.args.agent_num), self.cuda)
        # decoder transforms hidden states to action vector
        if self.args.continuous:
            self.action_mean = nn.Linear(self.args.hid_size, self.args.action_dim)
            # self.action_log_std = nn.Parameter(torch.zeros(1, self.args.action_dim))
        else:
            self.action_head = nn.Linear(self.args.hid_size, self.args.action_dim)
        # define communication inference
        # self.f_module = nn.Linear(self.args.hid_size, self.args.hid_size)
        self.f_modules = nn.ModuleList([nn.Linear(self.args.hid_size, self.args.hid_size) for _ in range(self.args.comm_iters)])
        # define communication encoder
        # self.C_module = nn.Linear(self.args.hid_size, self.args.hid_size)
        self.C_modules = nn.ModuleList([nn.Linear(self.args.hid_size, self.args.hid_size) for _ in range(self.args.comm_iters)])
        # if it is the skip connection then define another encoding transformation
        if self.args.skip_connection:
            self.E_module = nn.Linear(self.args.hid_size, self.args.hid_size)
            self.E_modules = nn.ModuleList([self.E_module for _ in range(self.args.comm_iters)])
        if self.args.training_strategy in ['reinforce', 'actor_critic']:
            # define value function
            self.value_head = nn.Linear(self.args.hid_size, 1)
        elif self.args.training_strategy in ['ddpg']:
            # define action value function
            self.value_head = nn.Linear(self.args.hid_size+self.args.action_dim, 1)

    def state_encoder(self, x):
        '''
        define a state encoder
        '''
        return torch.tanh(self.encoder(x))

    def policy(self, obs, info={}):
        '''
        define the action process of vanilla CommNet
        '''
        # get the batch size
        batch_size = obs.size()[0]
        # get the total number of agents including dead
        n = self.args.agent_num
        # encode observation
        if self.args.skip_connection:
            e = self.state_encoder(obs)
            h = torch.zeros_like(e)
        else:
            h = self.state_encoder(obs)
        # get the agent mask
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        # conduct the main process of communication
        for i in range(self.args.comm_iters):
            # shape = (batch_size, n, hid_size)->(batch_size, n, 1, hid_size)->(batch_size, n, n, hid_size)
            h_ = h.unsqueeze(-2).expand(batch_size, n, n, self.args.hid_size)
            # construct the communication mask
            mask = self.comm_mask.view(1, n, n) # shape = (1, n, n)
            mask = mask.expand(batch_size, n, n) # shape = (batch_size, n, n)
            mask = mask.unsqueeze(-1) # shape = (batch_size, n, n, 1)
            mask = mask.expand_as(h_) # shape = (batch_size, n, n, hid_size)
            # mask each agent itself (collect the hidden state of other agents)
            h_ = h_ * mask
            # mask the dead agent
            h_ = h_ * agent_mask * agent_mask.transpose(1, 2)
            # average the hidden state
            h_ = h_ / (num_agents_alive - 1) if num_agents_alive > 1 else torch.zeros_like(h_)
            # calculate the communication vector
            c = h_.sum(dim=1) if i != 0 else torch.zeros_like(h) # shape = (batch_size, n, hid_size)
            if self.args.skip_connection:
                # h_{j}^{i+1} = \sigma(H_j * h_j^{i+1} + C_j * c_j^{i+1} + E_{j} * e_j^{i+1})
                h = torch.tanh(sum([self.f_modules[i](h), self.C_modules[i](c), self.E_modules[i](e)]))
            else:
                # h_{j}^{i+1} = \sigma(H_j * h_j^{i+1} + C_j * c_j^{i+1})
                h = torch.tanh(sum([self.f_modules[i](h), self.C_modules[i](c)]))
        self.hidden = h
        # calculate the action vector (policy)
        if self.args.continuous:
            # shape = (batch_size, n, action_dim)
            action_mean = self.action_mean(h)
            # will be used later to sample
            action = action_mean
        else:
            # discrete actions, shape = (batch_size, n, action_type, action_num)
            action = self.action_head(h)
        return action

    def value(self, action):
        if self.args.training_strategy in ['ddpg']:
            return self.value_head(torch.cat((self.hidden, action), -1))
        else:
            return self.value_head(self.hidden)
