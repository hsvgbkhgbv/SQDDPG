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
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()

    def reload_params_to_target(self):
        self.target_net.action_dict.load_state_dict( self.action_dict.state_dict() )
        self.target_net.value_dict.load_state_dict( self.value_dict.state_dict() )

    def update_target(self):
        params_target_action = list(self.target_net.action_dict.parameters())
        params_behaviour_action = list(self.action_dict.parameters())
        for i in range(len(params_target_action)):
            params_target_action[i] = (1 - self.args.target_lr) * params_target_action[i] + self.args.target_lr * params_behaviour_action[i]
        params_target_value = list(self.target_net.value_dict.parameters())
        params_behaviour_value = list(self.value_dict.parameters())
        for i in range(len(params_target_value)):
            params_target_value[i] = (1 - self.args.target_lr) * params_target_value[i] + self.args.target_lr * params_behaviour_value[i]
        print ('traget net is updated!\n')

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
        batch_size = obs.size(0)
        act = act.contiguous().view( -1, np.prod(act.size()[1:]) ).unsqueeze(-2).expand(batch_size, self.n_, self.n_*self.act_dim)
        obs = obs.contiguous().view( -1, np.prod(obs.size()[1:]) )
        h = torch.relu( self.value_dict['layer_1']( torch.cat( (obs, act), dim=-1 ) ) )
        h = torch.relu( self.vaue_dict['layer_2'](h) )
        vs = self.value_dict['value_head'](h)
        return vs

    def td_lambda(self, state):
        z_v = cuda_wrapper(torch.zeros_like(state), self.cuda_)
        z_a = cuda_wrapper(torch.zeros_like(state), self.cuda_)

    def get_loss(self, batch):
        batch_size = len(batch.state)
        n = self.args.agent_num
        action_dim = self.args.action_dim
        # collect the transition data
        rewards, last_step, done, actions, state, next_state = unpack_data(self.args, batch)
        # construct computational graph
        action_out = self.policy(state)
        values = self.value(state, actions)
        if self.args.q_func:
            values = torch.sum(values*actions, dim=-1)
        values = values.contiguous().view(-1, n)
        next_action_out = self.target_net.policy(next_state)
        next_actions = select_action(self.args, next_action_out, status='train')
        next_values = self.target_net.value(next_state, next_actions)
        if self.args.q_func:
            next_values = torch.sum(next_values*next_actions, dim=-1)
        next_values = next_values.contiguous().view(-1, n)
        assert values.size() == next_values.size()
        deltas = rewards + self.args.gamma * next_values.detach() - values
        # calculate coma

    def init_hidden(self, batch_size):
        return cuda_wrapper(torch.zeros(batch_size*self.n_, self.hid_dim), self.cuda_)
