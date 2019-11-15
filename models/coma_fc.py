import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from collections import namedtuple



class COMAFC(Model):

    def __init__(self, args, target_net=None):
        super(COMAFC, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'last_step'))

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
            l1 = nn.Linear((self.n_+1)*self.obs_dim+(self.n_-1)*self.act_dim, self.hid_dim)
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
                value_dicts.append(nn.ModuleDict( {'layer_1': nn.Linear((self.n_+1)*self.obs_dim+(self.n_-1)*self.act_dim, self.hid_dim),\
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
        batch_size = obs.size(0)
        obs_own = obs.clone()
        obs = obs.unsqueeze(1).expand(batch_size, self.n_, self.n_, self.obs_dim) # shape = (b, n, o) -> (b, 1, n, o) -> (b, n, n, o)
        obs = obs.contiguous().view(batch_size, self.n_, -1) # shape = (b, n, o*n)
        inp = torch.cat((obs, obs_own), dim=-1) # shape = (b, n, o*n+o)
        values = []
        for i in range(self.n_):
            # other people actions 
            act_other = torch.cat((act[:,:i,:].view(batch_size,-1),act[:,i+1:,:].view(batch_size,-1)),dim=-1)
            h = torch.relu( self.value_dicts[i]['layer_1'](torch.cat((inp[:, i, :], act_other),dim=-1)) )
            h = torch.relu( self.value_dicts[i]['layer_2'](h) )
            v = self.value_dicts[i]['value_head'](h)
            values.append(v)
        values = torch.stack(values, dim=1)
        return values


    def get_loss(self, batch):
        batch_size = len(batch.state)
        rewards, last_step, done, actions, state, next_state = self.unpack_data(batch)
        action_out = self.policy(state) #  (b,n,a) action probability
        values = self.value(state, actions) # (b,n,a) action value
        baselines = torch.sum(values*torch.softmax(action_out, dim=-1), dim=-1)   # the only difference to ActorCritic is this  baseline (b,n)
        values = torch.sum(values*actions, dim=-1) # (b,n)
        if self.args.target:
            next_action_out = self.target_net.policy(next_state, last_act=actions)
        else:
            next_action_out = self.policy(next_state, last_act=actions)
        next_actions = select_action(self.args, next_action_out, status='train',  exploration=False)
        if self.args.target:
            next_values = self.target_net.value(next_state, next_actions)
        else:
            next_values = self.value(next_state, next_actions)
        next_values = torch.sum(next_values*next_actions, dim=-1) # b*n

        # calculate the advantages
        returns = cuda_wrapper(torch.zeros((batch_size, self.n_), dtype=torch.float), self.cuda_)
        assert values.size() == next_values.size()
        assert returns.size() == values.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_values[i].detach()
            else:
                next_return = next_values[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return

        # value loss
        deltas = returns - values
        value_loss = deltas.pow(2).mean(dim=0)

        # actio loss
        advantages = ( values - baselines ).detach() 
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        log_prob = multinomials_log_density(actions, action_out).contiguous().view(-1, self.n_)
        assert log_prob.size() == advantages.size()
        action_loss = - advantages * log_prob
        action_loss = action_loss.mean(dim=0)

        return action_loss, value_loss, action_out
