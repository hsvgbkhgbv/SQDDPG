import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model



class MADDPG(Model):

    def __init__(self, args):
        super(MADDPG, self).__init__(args)
        assert self.ts_ == 'ddpg'

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'layer_1': nn.ModuleList( [ nn.Linear(self.obs_dim, self.hid_dim) for _ in range(self.n_) ] ),\
                                           'layer_2': nn.ModuleList( [ nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_) ] ),\
                                           'action_head': nn.ModuleList( [ nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_) ] )
                                          }
                                        )

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict( {'layer_1': nn.ModuleList( [ nn.Linear( (self.obs_dim+self.act_dim)*self.n_, self.hid_dim ) for _ in range(self.n_) ] ),\
                                          'layer_2': nn.ModuleList( [ nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_) ] ),\
                                          'value_head': nn.ModuleList( [ nn.Linear(self.hid_dim, 1) for _ in range(self.n_) ] )
                                         }
                                       )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def policy(self, obs, info={}, stat={}):
        actions = []
        for i in range(self.n_):
            h = torch.relu( self.action_dict['layer_1'][i](obs[:, i, :]) )
            h = torch.relu( self.action_dict['layer_2'][i](h) )
            a = self.action_dict['action_head'][i](h)
            actions.append(a)
        actions = torch.stack(actions, dim=1)
        return actions

    def value(self, obs, act):
        values = []
        for i in range(self.n_):
            h = torch.relu( self.value_dict['layer_1'][i]( torch.cat( ( obs.contiguous().view( -1, np.prod(obs.size()[1:]) ), act.contiguous().view( -1, np.prod(act.size()[1:]) ) ), dim=-1 ) ) )
            h = torch.relu( self.value_dict['layer_2'][i](h) )
            v = self.value_dict['value_head'][i](h)
            values.append(v)
        values = torch.stack(values, dim=1)
        return values
