import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from collections import namedtuple
from learning_algorithms.actor_critic import *

# Mean Field Actor Critic
class MFAC(Model):

    def __init__(self, args, target_net=None):
        super(MFAC, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.Transition = namedtuple('Transition', ('state', 'action','reward', 'next_state', 'done', 'last_step'))
        self.rl = ActorCritic(self.args)

    def construct_policy_net(self):
        # TODO: fix policy params update
        action_dicts = []
        if self.args.shared_parameters:
            l1 = nn.Linear(self.obs_dim, self.hid_dim)
            l2 = nn.Linear(self.hid_dim, self.hid_dim)
            a = nn.Linear(self.hid_dim, self.act_dim)
            for i in range(self.n_):
                action_dicts.append(nn.ModuleDict( {'layer_1': l1,\
                                                    'layer_2': l2,\
                                                    'action_head': a
                                                    }
                                                 )
                                   )
        else:
            for i in range(self.n_):
                action_dicts.append(nn.ModuleDict( {'layer_1': nn.Linear(self.obs_dim, self.hid_dim),\
                                                    'layer_2': nn.Linear(self.hid_dim, self.hid_dim),\
                                                    'action_head': nn.Linear(self.hid_dim, self.act_dim)
                                                    }
                                                  )
                                   )
        self.action_dicts = nn.ModuleList(action_dicts)

    def construct_value_net(self):
        # TODO: policy params update
        value_dicts = []
        if self.args.shared_parameters:
            l1 = nn.Linear(self.obs_dim*self.n_+ self.act_dim, self.hid_dim)
            l2 = nn.Linear(self.hid_dim, self.hid_dim)
            v = nn.Linear(self.hid_dim, self.act_dim)
            for i in range(self.n_):
                value_dicts.append(nn.ModuleDict( {'layer_1': l1,\
                                                   'layer_2': l2,\
                                                   'value_head': v
                                                  }
                                                )
                                  )
        else:
            for i in range(self.n_):
                value_dicts.append(nn.ModuleDict( {'layer_1': nn.Linear(self.obs_dim*self.n_+self.act_dim, self.hid_dim),\
                                                   'layer_2': nn.Linear(self.hid_dim, self.hid_dim),\
                                                   'value_head': nn.Linear(self.hid_dim, self.act_dim)
                                                  }
                                                )
                                  )
        self.value_dicts = nn.ModuleList(value_dicts)

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    
    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        # TODO: policy params update
        actions = []
        for i in range(self.n_):
            h = torch.relu( self.action_dicts[i]['layer_1'](obs[:, i, :]) )
            h = torch.relu( self.action_dicts[i]['layer_2'](h) )
            a = self.action_dicts[i]['action_head'](h)
            actions.append(a)
        actions = torch.stack(actions, dim=1)
        return actions
        
    def value(self, obs, act):
        # TODO: policy params update
        batch_size = obs.size(0)
        # expand obs
        obs = obs.unsqueeze(1).expand(batch_size, self.n_, self.n_, self.obs_dim).contiguous().view(batch_size, self.n_, -1) # shape = (b, n, o) -> (b, 1, n, o) -> (b, n, n, o) -> (b, n, n*o)
        # calculate mean_act: MF neighbours include itself
        mean_act = torch.mean(act, dim=1, keepdim=True).repeat(1, self.n_,  1) # shape = (b, n, a) -> (b, 1, a) -> (b, n, a)
        mean_act = ( mean_act * self.n_ - act ) / max(1, self.n_-1)
        inp = torch.cat((obs, mean_act),dim=-1) # shape = (b, n, o*n+a) 
        values = []
        for i in range(self.n_):
            h = torch.relu( self.value_dicts[i]['layer_1'](inp[:, i, :]) )
            h = torch.relu( self.value_dicts[i]['layer_2'](h) )
            v = self.value_dicts[i]['value_head'](h)
            values.append(v)
        values = torch.stack(values, dim=1)
        return values

    def get_loss(self, batch):
        action_loss, value_loss, log_p_a = self.rl.get_loss(batch, self, self.target_net)
        return action_loss, value_loss, log_p_a

