import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from learning_algorithms.reinforce import *



class CommNet(Model):

    def __init__(self, args):
        super(CommNet, self).__init__(args)
        self.comm_iters = self.args.comm_iters
        self.rl = REINFORCE(self.args)
        if self.comm_iters == 0:
            raise RuntimeError('Please guarantee the comm iters is at least greater equal to 1.')
        elif self.comm_iters < 2:
            raise RuntimeError('Please use IndependentCommNet if the comm iters is set to 1.')
        self.construct_model()
        self.apply(self.init_weights)

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'encoder': nn.Linear(self.obs_dim, self.hid_dim),\
                                           'f_module': nn.Linear(self.hid_dim, self.hid_dim),\
                                           'c_module': nn.Linear(self.hid_dim, self.hid_dim),\
                                           'action_head': nn.Linear(self.hid_dim, self.act_dim)
                                          }
                                        )
        self.action_dict['f_modules'] = nn.ModuleList( [ self.action_dict['f_module'] for _ in range(self.comm_iters) ] )
        self.action_dict['c_modules'] = nn.ModuleList( [ self.action_dict['c_module'] for _ in range(self.comm_iters) ] )
        if self.args.skip_connection:
            self.action_dict['e_module'] = nn.Linear(self.hid_dim, self.hid_dim)
            self.action_dict['e_modules'] = nn.ModuleList( [ self.action_dict['e_module'] for _ in range(self.comm_iters) ] )

    def construct_model(self):
        self.comm_mask = cuda_wrapper(torch.ones(self.n_, self.n_) - torch.eye(self.n_, self.n_), self.cuda_)
        self.construct_value_net()
        self.construct_policy_net()

    def policy(self, obs, info={}, stat={}):
        # get the batch size
        batch_size = obs.size(0)
        # encode observation
        if self.args.skip_connection:
            e = torch.relu( self.action_dict['encoder'](obs) )
            h = torch.zeros_like(e)
        else:
            h = torch.tanh( self.action_dict['encoder'](obs) )
        # get the agent mask
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        # conduct the main process of communication
        for i in range(self.comm_iters):
            # shape = (batch_size, n, hid_size)->(batch_size, n, 1, hid_size)->(batch_size, n, n, hid_size)
            h_ = h.unsqueeze(-2).expand(batch_size, self.n_, self.n_, self.hid_dim)
            # construct the communication mask
            mask = self.comm_mask.view(1, self.n_, self.n_) # shape = (1, n, n)
            mask = mask.expand(batch_size, self.n_, self.n_) # shape = (batch_size, n, n)
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
                h = torch.tanh( sum( [ self.action_dict['f_modules'][i](h), self.action_dict['c_modules'][i](c), self.action_dict['e_modules'][i](e) ] ) )
            else:
                # h_{j}^{i+1} = \sigma(H_j * h_j^{i+1} + C_j * c_j^{i+1})
                h = torch.tanh( sum( [ self.action_dict['f_modules'][i](h), self.action_dict['c_modules'][i](c) ] ) )
        # calculate the action vector (policy)
        action = self.action_dict['action_head'](h)
        return action

    def get_loss(self, batch):
        action_loss, value_loss, log_p_a = self.rl.get_loss(batch, self)
        return action_loss, value_loss, log_p_a



class IndependentCommNet(CommNet):
    def __init__(self, args):
        super(IndependentCommNet, self).__init__(args)
        self.comm_iters = 1
