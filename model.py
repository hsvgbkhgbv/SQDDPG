import torch
import torch.nn as nn
from util import *



class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.cuda_ = torch.cuda.is_available() and self.args.cuda
        self.construct_model()
        self.apply(self.init_weights)

    def construct_model(self):
        raise NotImplementedError()

    def state_encoder(self, x):
        raise NotImplementedError()

    def get_agent_mask(self, batch_size, info):
        '''
        define the getter of agent mask to confirm the living agent
        '''
        n = self.args.agent_num
        with torch.no_grad():
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

    def policy(self, obs, info={}):
        raise NotImplementedError()

    def value(self):
        raise NotImplementedError()
    
    def construct_policy_net(self):
        raise NotImplementedError()
    
    def construct_value_net(self):
        raise NotImplementedError()
        
    def init_weights(self, m):
        '''
        initialize the weights of parameters
        '''
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.args.init_std)
