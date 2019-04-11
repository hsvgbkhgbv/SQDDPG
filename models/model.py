import torch
import torch.nn as nn
import numpy as np
from utilities.util import *



class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.cuda_ = torch.cuda.is_available() and self.args.cuda
        self.ts_ = self.args.training_strategy
        self.n_ = self.args.agent_num
        self.hid_dim = self.args.hid_size
        self.obs_dim = self.args.obs_size
        self.act_dim = self.args.action_dim
        self.comm_iters = self.args.comm_iters
        self.construct_model()
        self.apply(self.init_weights)

    def construct_model(self):
        raise NotImplementedError()

    def get_agent_mask(self, batch_size, info):
        '''
        define the getter of agent mask to confirm the living agent
        '''
        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(self.n_)
            num_agents_alive = self.n_
        # shape = (1, 1, n)
        agent_mask = agent_mask.view(1, 1, self.n_)
        # shape = (batch_size, n ,n, 1)
        agent_mask = cuda_wrapper(agent_mask.expand(batch_size, self.n_, self.n_).unsqueeze(-1), self.cuda_)
        return num_agents_alive, agent_mask

    def policy(self, obs, info={}, stat={}):
        raise NotImplementedError()

    def value(self, obs, act):
        if self.args.training_strategy in ['ddpg']:
            h = self.value_dict['value_body'](torch.cat((obs, act), -1))
        else:
            h = self.value_dict['value_body'](obs)
        h = torch.relu(h)
        v = self.value_dict['value_head'](h)
        return v

    def construct_policy_net(self):
        raise NotImplementedError()

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict()
        if self.ts_ in ['ddpg']:
            self.value_dict['value_body'] = nn.Linear(self.obs_dim+self.act_dim, self.hid_dim)
            self.value_dict['value_head'] = nn.Linear(self.hid_dim, 1)
        else:
            self.value_dict['value_body'] = nn.Linear(self.obs_dim, self.hid_dim)
            if self.args.q_func:
                assert self.args.training_strategy not in ['reinforce']
                self.value_dict['value_head'] = nn.Linear(self.hid_dim, self.act_dim)
            else:
                self.value_dict['value_head'] = nn.Linear(self.hid_dim, 1)

    def init_weights(self, m):
        '''
        initialize the weights of parameters
        '''
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.args.init_std)
