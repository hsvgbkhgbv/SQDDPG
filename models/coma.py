import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model



import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model



class COMA(Model):

    def __init__(self, args):
        super(COMA, self).__init__(args)
        assert self.ts_ == 'actor_critic'
        self.construct_model()

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'observation': nn.Linear(self.obs_dim, self.hid_dim),\
                                           'gru_layer': nn.GRUCell(self.hid_dim+self.act_dim, self.hid_dim),\
                                           'action_head': nn.Linear(self.hid_dim, self.act_dim)
                                          }
                                        )

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict( {'layer_1': nn.Linear( (self.obs_dim+self.act_dim)*self.n_, self.hid_dim ),\
                                          'layer_2': nn.Linear( self.hid_dim, self.hid_dim ),\
                                          'value_head': nn.Linear(self.hid_dim, self.act_dim)
                                         }
                                       )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def policy(self, obs, info={}, stat={}):
        batch_size = obs.size(0)
        last_act = info['last_actions'] # shape=(batch_size, n, act_dim)
        h = torch.relu( self.action_dict['observation'](obs) )
        h = torch.cat((h, last_act), dim=-1) # shape=(batch_size, n, act_dim+hid_dim)
        h = h.contiguous().view(batch_size*self.n_, self.act_dim+self.obs_dim) # shape=(batch_size*n, act_dim+hid_dim)
        h = torch.relu( self.action_dict['gru_layer'](h) )
        as = self.action_dict['action_head'](h)
        return as

    def value(self, obs, act):
            h = torch.relu( self.value_dict['layer_1']( torch.cat( ( obs.contiguous().view( -1, np.prod(obs.size()[1:]) ), act.contiguous().view( -1, np.prod(act.size()[1:]) ) ), dim=-1 ) ) )
            h = torch.relu( self.vaue_dict['layer_2'](h) )
            vs = self.value_dict['value_head'](h)
        return vs

    def get_loss(self):
        pass
    
    def init_hidden(self, batch_size):
        return cuda_wrapper(torch.zeros(batch_size*self.n_, self.hid_dim), self.cuda_)
