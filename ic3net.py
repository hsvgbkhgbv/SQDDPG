import torch
import torch.nn as nn
import numpy as np
from model import Model
from util import *



class IC3Net(Model):

    def __init__(self, args):
        super(IC3Net, self).__init__(args)

    def construct_model(self):
        '''
        define the model of vanilla CommNet
        '''
        # encoder transforms observation to latent variables
        self.encoder = nn.Linear(self.args.obs_size, self.args.hid_size)
        # communication mask where the diagnal should be 0
        self.comm_mask = cuda_wrapper(torch.ones(self.args.agent_num, self.args.agent_num) - torch.eye(self.args.agent_num, self.args.agent_num), self.cuda_)
        # decoder transforms hidden states to action vector
        if self.args.continuous:
            self.action_mean = nn.Linear(self.args.hid_size, self.args.action_dim)
            # self.action_log_std = nn.Parameter(torch.zeros(1, self.args.action_dim))
        else:
            self.action_head = nn.Linear(self.args.hid_size, self.args.action_dim)
        # define communication inference
        self.f_module = nn.LSTMCell(self.args.hid_size, self.args.hid_size)
        # define the gate function
        # self.g_module = nn.Linear(self.args.hid_size, self.args.agent_num)
        self.g_modules = nn.ModuleList([nn.Linear(self.args.hid_size, self.args.agent_num) for _ in range(self.args.comm_iters)])
        # define value function or the action value function
        if self.args.training_strategy in ['reinforce', 'actor_critic']:
            self.value_head = nn.Linear(self.args.hid_size, 1)
        elif self.args.training_strategy in ['ddpg']:
            self.value_head = nn.Linear(self.args.hid_size+self.args.action_dim, 1)

    def state_encoder(self, x):
        '''
        define a state encoder
        '''
        return torch.tanh(self.encoder(x))

    def policy(self, obs, info={}):
        '''
        define the action process of IC3Net
        '''
        # get the batch_size
        batch_size = obs.size(0)
        # get the total number of agents including dead
        n = self.args.agent_num
        # encode observation
        e = self.state_encoder(obs)
        # get the initial state
        h, cell = self.init_hidden(batch_size)
        # get the agent mask
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        # conduct the main process of communication
        for i in range(self.args.comm_iters):
            h_ = h.contiguous().view(batch_size, n, self.args.hid_size)
            # define the gate function
            gate_ = torch.sigmoid(self.g_modules[i](h_))
            gate = torch.round(gate_)
            # shape = (batch_size, n, hid_size)->(batch_size, n, 1, hid_size)->(batch_size, n, n, hid_size)
            h_ = h_.unsqueeze(-2).expand(batch_size, n, n, self.args.hid_size)
            # construct the communication mask
            mask = self.comm_mask.contiguous().view(1, n, n) # shape = (1, n, n)
            mask = mask.expand(batch_size, n, n) # shape = (batch_size, n, n)
            mask = mask.unsqueeze(-1) # shape = (batch_size, n, n, 1)
            mask = mask.expand_as(h_) # shape = (batch_size, n, n, hid_size)
            # construct the commnication gate
            gate = gate.contiguous().view(batch_size, n, n) # shape = (batch_size, n, n)
            gate = gate.unsqueeze(-1) # shape = (batch_size, n, n, 1)
            gate = gate.expand_as(h_) # shape = (batch_size, n, n, hid_size)
            # mask each agent itself (collect the hidden state of other agents)
            h_ = h_ * gate * mask
            # mask the dead agent
            h_ = h_ * agent_mask * agent_mask.transpose(1, 2)
            # average the hidden state
            if num_agents_alive > 1: h_ = h_ / (num_agents_alive - 1)
            # calculate the communication vector
            c = h_.sum(dim=1) # shape = (batch_size, n, hid_size)
            inp = e + c
            inp = inp.contiguous().view(batch_size*n, self.args.hid_size)
            # f_moudle
            h, cell = self.f_module(inp, (h, cell))
        h = h.contiguous().view(batch_size, n, self.args.hid_size)
        self.hidden = h
        # calculate the action vector (policy)
        if self.args.continuous:
            # shape = (batch_size, n, action_dim)
            action_mean = self.action_mean(h)
            # will be used later to sample
            action = action_mean
        else:
            # discrete actions, shape = (batch_size, n, action_dim)
            action = self.action_head(h)
        return action

    def value(self, action):
        if self.args.training_strategy in ['ddpg']:
            return self.value_head(torch.cat((self.hidden, action), -1))
        else:
            return self.value_head(self.hidden)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple((torch.zeros(batch_size * self.args.agent_num, self.args.hid_size),
                      torch.zeros(batch_size * self.args.agent_num, self.args.hid_size)))
