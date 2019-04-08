import torch
import torch.nn as nn
import numpy as np
from model import Model
from util import *



class IC3Net(Model):

    def __init__(self, args):
        super(IC3Net, self).__init__(args)

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'encoder': nn.Linear(self.obs_dim, self.hid_dim),\
                                           'f_module': nn.LSTMCell(self.hid_dim, self.hid_dim),\
                                           'g_module': nn.Linear(self.hid_dim, self.n_),\
                                           'action_head': nn.Linear(self.hid_dim, self.act_dim)
                                          } )
        self.action_dict['g_modules'] = nn.ModuleList( [ self.action_dict['g_module'] for _ in range(self.comm_iters) ] )

    def construct_model(self):
        self.comm_mask = cuda_wrapper(torch.ones(self.n_, self.n_) - torch.eye(self.n_, self.n_), self.cuda_)
        self.construct_value_net()
        self.construct_policy_net()

    def policy(self, obs, info={}):
        batch_size = obs.size(0)
        # encode observation
        e = torch.relu(self.action_dict['encoder'](obs))
        # get the initial state
        h, cell = self.init_hidden(batch_size)
        # get the agent mask
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        # conduct the main process of communication
        for i in range(self.comm_iters):
            h_ = h.contiguous().view(batch_size, self.n_, self.hid_dim)
            # define the gate function
            gate_ = torch.sigmoid(self.action_dict['g_modules'][i](h_))
            gate = torch.round(gate_)
            # shape = (batch_size, n, hid_size)->(batch_size, n, 1, hid_size)->(batch_size, n, n, hid_size)
            h_ = h_.unsqueeze(-2).expand(batch_size, self.n_, self.n_, self.hid_dim)
            # construct the communication mask
            mask = self.comm_mask.contiguous().view(1, self.n_, self.n_) # shape = (1, n, n)
            mask = mask.expand(batch_size, self.n_, self.n_) # shape = (batch_size, n, n)
            mask = mask.unsqueeze(-1) # shape = (batch_size, n, n, 1)
            mask = mask.expand_as(h_) # shape = (batch_size, n, n, hid_size)
            # construct the commnication gate
            gate = gate.contiguous().view(batch_size, self.n_, self.n_) # shape = (batch_size, n, n)
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
            inp = inp.contiguous().view(batch_size*self.n_, self.hid_dim)
            # f_moudle
            h, cell = self.action_dict['f_module'](inp, (h, cell))
        h = h.contiguous().view(batch_size, self.n_, self.hid_dim)
        # calculate the action vector (policy)
        action = self.action_dict['action_head'](h)
        return action

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return (cuda_wrapper(torch.zeros(batch_size * self.n_, self.hid_dim), self.cuda_),
                cuda_wrapper(torch.zeros(batch_size * self.n_, self.hid_dim), self.cuda_))
