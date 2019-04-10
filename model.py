import torch
import torch.nn as nn
from util import *



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
        n = self.args.agent_num
        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n
        # shape = (1, 1, n)
        agent_mask = agent_mask.view(1, 1, n)
        # shape = (batch_size, n ,n, 1)
        agent_mask = cuda_wrapper(agent_mask.expand(batch_size, n, n).unsqueeze(-1), self.cuda_)
        return num_agents_alive, agent_mask

    def policy(self, obs, info={}, stat={}):
        raise NotImplementedError()

    def value(self, obs, act):
        if self.args.training_strategy in ['ddpg']:
            h = self.value_dict['value_body'](torch.cat((obs, act), -1))
            h = torch.relu(h)
            h = self.value_dict['value_head'](h)
        else:
            h = self.value_dict['value_body'](obs)
            h = torch.relu(h)
            h = self.value_dict['value_head'](h)
        return h

    def construct_policy_net(self):
        raise NotImplementedError()

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict()
        if self.ts_ in ['ddpg']:
            self.value_dict['value_body'] = nn.Linear(self.obs_dim+self.act_dim, self.hid_dim)
            self.value_dict['value_head'] = nn.Linear(self.hid_dim, 1)
        else:
            self.value_dict['value_body'] = nn.Linear(self.obs_dim, self.hid_dim)
            self.value_dict['value_head'] = nn.Linear(self.hid_dim, 1)

    def init_weights(self, m):
        '''
        initialize the weights of parameters
        '''
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.args.init_std)
