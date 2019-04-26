import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model



class COMA(Model):

    def __init__(self, args, target_net=None):
        super(COMA, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        if self.args.epsilon_softmax:
            self.eps_delta = (args.softmax_eps_init - args.softmax_eps_end) / (args.epoch_size*args.train_epoch_num)
            self.eps = args.softmax_eps_init
        self.gru_hids = []

    def update_eps(self):
        self.eps -= self.eps_delta

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
        # print ('traget net is updated!\n')

    def construct_policy_net(self):
        # self.action_dict = nn.ModuleDict( {'transform': nn.Linear(self.obs_dim+self.act_dim, self.hid_dim),\
        #                                    'gru_layer': nn.GRUCell(self.hid_dim, self.hid_dim),\
        #                                    'action_head': nn.Linear(self.hid_dim, self.act_dim)
        #                                   }
        #                                 )
        self.action_dict = nn.ModuleDict( {'transform': nn.ModuleList([nn.Linear(self.obs_dim+self.act_dim, self.hid_dim) for _ in range(self.n_)]),\
                                           'gru_layer': nn.ModuleList([nn.GRUCell(self.hid_dim, self.hid_dim) for _ in range(self.n_)]),\
                                           'action_head': nn.ModuleList([nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_)])
                                          }
                                        )
        # self.action_dict = nn.ModuleDict( {'transform': nn.ModuleList([nn.Linear(self.obs_dim+self.act_dim, self.hid_dim) for _ in range(self.n_)]),\
        #                                    'gru_layer': nn.ModuleList([nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_)]),\
        #                                    'action_head': nn.ModuleList([nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_)])
        #                                   }
        #                                 )

    def construct_value_net(self):
        # self.value_dict = nn.ModuleDict( {'layer_1': nn.Linear( self.obs_dim+self.act_dim*(self.n_-1)+self.act_dim*self.n_, self.hid_dim ),\
        #                                   'layer_2': nn.Linear(self.hid_dim, self.hid_dim),\
        #                                   'value_head': nn.Linear(self.hid_dim, self.act_dim)
        #                                  }
        #                                )
        self.value_dict = nn.ModuleDict( {'layer_1': nn.ModuleList([nn.Linear( self.obs_dim+self.act_dim*(self.n_-1)+self.act_dim*self.n_, self.hid_dim ) for _ in range(self.n_)]),\
                                          'layer_2': nn.ModuleList([nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_)]),\
                                          'value_head': nn.ModuleList([nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_)])
                                         }
                                       )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def policy(self, obs, last_act, info={}, stat={}):
        batch_size = obs.size(0)
        if not info.get('get_episode', False):
            if info.get('batch_train_curr', False):
                gru_hids = []
                for i in range(self.args.epoch_size):
                    gru_hids.extend( self.gru_hids[(self.args.max_steps+1)*i:(self.args.max_steps+1)*(i+1)-1] )
                self.gru_hid = cuda_wrapper(torch.cat(gru_hids, dim=1), cuda=self.cuda_)
            else:
                gru_hids = []
                for i in range(self.args.epoch_size):
                    gru_hids.extend( self.gru_hids[(self.args.max_steps+1)*i+1:(self.args.max_steps+1)*(i+1)] )
                self.gru_hid = cuda_wrapper(torch.cat(gru_hids, dim=1), cuda=self.cuda_)
        actions = []
        for i in range(self.n_):
            h = torch.cat( (obs[:, i, :], last_act[:, i, :]), dim=-1 ) # shape=(batch_size, n, act_dim+hid_dim)
            h = h.contiguous().view(batch_size, self.obs_dim+self.act_dim) # shape=(batch_size*n, act_dim+hid_dim)
            h = torch.relu( self.action_dict['transform'][i](h) )
            h = self.action_dict['gru_layer'][i](h, self.gru_hid[i])
            # h = self.action_dict['gru_layer'][i](h)
            if info.get('get_episode', False):
                self.gru_hid[i] = h
            h = torch.relu(h)
            h = h.contiguous().view(batch_size, self.hid_dim)
            a = self.action_dict['action_head'][i](h)
            actions.append(a)
        a = torch.stack(actions, dim=1)
        return a

    def value(self, obs, act):
        batch_size = obs.size(0)
        act, last_act = act
        act = act.contiguous().view( -1, np.prod(act.size()[1:]) ).unsqueeze(-2).expand(batch_size, self.n_, self.n_*self.act_dim)
        values = []
        for i in range(self.n_):
            h = torch.relu( self.value_dict['layer_1'][i]( torch.cat( (obs[:, i, :], act[:, i, :i*self.act_dim], act[:, i, (i+1)*self.act_dim:], last_act.contiguous().view(-1, self.n_*self.act_dim)), dim=-1 ) ) )
            # h = torch.relu( self.value_dict['layer_1'][i]( torch.cat( (obs[:, i, :], act[:, i, :i*self.act_dim], act[:, i, (i+1)*self.act_dim:]), dim=-1 ) ) )
            h = torch.relu( self.value_dict['layer_2'][i](h) )
            v = self.value_dict['value_head'][i](h)
            values.append(v)
        values = torch.stack(values, dim=1)
        return values

    def get_loss(self, batch):
        info = {'softmax_eps': self.eps} if self.args.epsilon_softmax else {}
        info['batch_train_curr'] = True
        batch_size = len(batch.state)
        n = self.args.agent_num
        action_dim = self.args.action_dim
        # collect the transition data
        rewards, last_step, done, actions, last_actions, state, next_state = unpack_data(self.args, batch)
        # construct computational graph
        action_out = self.policy(state, last_actions, info=info)
        values_ = self.value( state, (actions, last_actions) )
        if self.args.q_func:
            values = torch.sum(values_*actions, dim=-1)
        values = values.contiguous().view(-1, n)
        info['batch_train_curr'] = False
        self.target_net.gru_hids = self.gru_hids
        next_action_out = self.target_net.policy(next_state, actions, info=info)
        next_actions = select_action(self.args, next_action_out, status='train', info=info)
        next_values = self.target_net.value( next_state, (next_actions, actions) )
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        if self.args.q_func:
            next_values = torch.sum(next_values*next_actions, dim=-1)
        next_values = next_values.contiguous().view(-1, n)
        # n-step TD estimate
        assert values.size() == next_values.size()
        assert returns.size() == rewards.size()
        returns = n_step(rewards, last_step, done, next_values, returns, self.args)
        # calculate coma
        advantages = ( values - torch.sum(values_*torch.softmax(action_out, dim=-1), dim=-1) ).detach()
        log_p_a = action_out
        log_prob = multinomials_log_density(actions, log_p_a).contiguous().view(-1, 1)
        advantages = advantages.contiguous().view(-1, 1)
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        assert log_prob.size() == advantages.size()
        action_loss = - advantages * log_prob
        action_loss = action_loss.sum() / batch_size
        # value_obj = - (returns - values.detach()) * td_lambda(values, self.args)
        # value_loss = value_obj.view(-1).sum() / batch_size
        value_loss = values.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, log_p_a

    def init_hidden(self, batch_size):
        self.gru_hid = cuda_wrapper(torch.zeros(self.n_, batch_size, self.hid_dim), self.cuda_)

    def clean_hidden(self):
        self.gru_hids = []

    def add_hidden(self):
        self.gru_hids.append(self.gru_hid.detach())
