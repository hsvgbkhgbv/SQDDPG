import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from learning_algorithms.reinforce import *



class IC3Net(Model):

    def __init__(self, args):
        super(IC3Net, self).__init__(args)
        self.rl = REINFORCE(self.args)
        self.construct_model()
        self.apply(self.init_weights)

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'encoder': nn.Linear(self.obs_dim, self.hid_dim),\
                                           'f_module': nn.LSTMCell(self.hid_dim, self.hid_dim),\
                                           'action_head': nn.Linear(self.hid_dim, self.act_dim)
                                          }
                                        )
        self.gate_dict = nn.ModuleDict( {'g_module': nn.Linear(self.hid_dim, 2)} )

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict()
        self.value_dict['value_body'] = nn.Linear(self.obs_dim, self.hid_dim)
        self.value_dict['value_head'] = nn.Linear(self.hid_dim, 1)

    def construct_model(self):
        self.comm_mask = cuda_wrapper(torch.ones(self.n_, self.n_) - torch.eye(self.n_, self.n_), self.cuda_)
        self.construct_value_net()
        self.construct_policy_net()

    def gate(self, batch_size, h):
        gate = torch.softmax( self.gate_dict['g_module'](h), dim=-1 ) # shape = (batch_size, n)
        return gate

    def policy(self, obs, last_act=None, last_hid=None, info={}, stat={}):
        batch_size = obs.size(0)
        # encode observation
        e = torch.relu(self.action_dict['encoder'](obs))
        # get the initial state
        if info.get('start', False):
            h, cell = self.init_hidden(batch_size)
        # get the agent mask
        # num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        # conduct the main process of communication
        h_ = h.contiguous().view(batch_size, self.n_, self.hid_dim)
        # define the gate function
        gate_ = self.gate(batch_size, h_).detach()
        gate_ = torch.argmin(gate_, keepdim=True) # act0: comm, act1: not comm
        # shape = (batch_size, n, hid_size)->(batch_size, n, 1, hid_size)->(batch_size, n, n, hid_size)
        h_ = h_.unsqueeze(-2).expand(batch_size, self.n_, self.n_, self.hid_dim)
        # construct the communication mask
        mask = self.comm_mask.unsqueeze(0) # shape = (1, n, n)
        mask = mask.expand(batch_size, self.n_, self.n_) # shape = (batch_size, n, n)
        mask = mask.unsqueeze(-1) # shape = (batch_size, n, n, 1)
        mask = mask.expand_as(h_) # shape = (batch_size, n, n, hid_size)
        # construct the commnication gate
        gate = gate_.unsqueeze(-1) # shape = (batch_size, n, n, 1)
        gate = gate.expand_as(h_) # shape = (batch_size, n, n, hid_size)
        # mask each agent itself (collect the hidden state of other agents)
        h_ = h_ * gate * mask
        # mask the dead agent
        # h_ = h_ * agent_mask * agent_mask.transpose(1, 2)
        # average the hidden state
        # if num_agents_alive > 1: h_ = h_ / (num_agents_alive - 1)
        h_ = h_ / (self.n_ - 1)
        # calculate the communication vector
        c = h_.sum(dim=2) # shape = (batch_size, n, hid_size)
        inp = e + c
        inp = inp.contiguous().view(batch_size*self.n_, self.hid_dim)
        # f_moudle
        h, cell = self.action_dict['f_module'](inp, (h, cell))
        h = h.contiguous().view(batch_size, self.n_, self.hid_dim)
        # calculate the action vector (policy)
        action = self.action_dict['action_head'](h)
        if batch_size == 1:
            stat['comm_gate'] = gate_.detach().cpu().numpy()
        return action

    def value(self, obs, act=None):
        h = self.value_dict['value_body'](obs)
        h = torch.relu(h)
        v = self.value_dict['value_head'](h)
        return v

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return (cuda_wrapper(torch.zeros(batch_size * self.n_, self.hid_dim), self.cuda_),
                cuda_wrapper(torch.zeros(batch_size * self.n_, self.hid_dim), self.cuda_))

    def get_loss(self, batch):
        action_loss, value_loss, log_p_a = self.rl.get_loss(batch, self)
        return action_loss, value_loss, log_p_a



class IndependentIC3Net(IC3Net):

    def __init__(self, args):
        super(IndependentIC3Net, self).__init__(args)

    def identifier(self):
        if self.comm_iters == 0:
            raise RuntimeError('Please guarantee the comm iters is at least greater equal to 1.')
        elif self.comm_iters > 1:
            raise RuntimeError('Please use IC3Net if the comm iters is set to the value greater than 1.')
