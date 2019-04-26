import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from learning_algorithms.q_learning import *
from learning_algorithms.actor_critic import *



class MF(Model):

    def __init__(self, args, target_net=None):
        super(MF, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()

    def reload_params_to_target(self):
        raise NotImplementedError()

    def update_target_action(self):
        params_target_action = list(self.target_net.action_dict.parameters())
        params_behaviour_action = list(self.action_dict.parameters())
        for i in range(len(params_target_action)):
            params_target_action[i] = (1 - self.args.target_lr) * params_target_action[i] + self.args.target_lr * params_behaviour_action[i]

    def update_target_value(self):
        params_target_value = list(self.target_net.value_dict.parameters())
        params_behaviour_value = list(self.value_dict.parameters())
        for i in range(len(params_target_value)):
            params_target_value[i] = (1 - self.args.target_lr) * params_target_value[i] + self.args.target_lr * params_behaviour_value[i]

    def update_target(self):
        self.update_target_action()
        self.update_target_value()
        # print ('traget net is updated!\n')

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def get_loss(self, batch):
        raise NotImplementedError()



class MFQ(MF):

    def __init__(self, args, target_net=None):
        super(MFQ, self).__init__(args, target_net)

    def reload_params_to_target(self):
        self.target_net.value_dict.load_state_dict( self.value_dict.state_dict() )

    def update_target_action(self):
        pass

    def construct_policy_net(self):
        pass

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict( {'layer_1': nn.ModuleList( [ nn.Linear(self.obs_dim+self.act_dim, self.hid_dim) for _ in range(self.n_) ] ),\
                                          'layer_2': nn.ModuleList( [ nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_) ] ),\
                                          'value_head': nn.ModuleList( [ nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_) ] )
                                         }
                                       )

    def policy(self, obs, last_act, info={}, stat={}):
        pass

    def value(self, obs, act, info={}, stat={}):
        batch_size = obs.size(0)
        values = []
        for i in range(self.n_):
            act_mean = torch.mean(torch.cat( (act[:, :i, :], act[:, (i+1):, :]), dim=1 ), dim=1)
            h = torch.relu( self.value_dict['layer_1'][i]( torch.cat( (obs[:, i, :], act_mean), dim=-1 ) ) )
            # h = torch.relu( self.value_dict['layer_1'][i](obs[:, i, :]) )
            h = torch.relu( self.value_dict['layer_2'][i](h) )
            v = self.value_dict['value_head'][i](h)
            values.append(v)
        values = torch.stack(values, dim=1)
        return values

    def get_loss(self, batch):
        batch_size = len(batch.state)
        n = self.args.agent_num
        action_dim = self.args.action_dim
        # collect the transition data
        rewards, last_step, done, actions, last_actions, state, next_state = unpack_data(self.args, batch)
        # construct the computational graph
        values = self.value(state, last_actions)
        values = torch.sum(values*actions, dim=-1)
        values = values.contiguous().view(-1, n)
        next_values = self.target_net.value(next_state, actions)
        next_values = torch.sum(torch.softmax(next_values/0.01, dim=-1)*next_values, dim=-1)
        next_values = next_values.contiguous().view(-1, n)
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        # calculate the advantages
        assert values.size() == next_values.size()
        assert returns.size() == values.size()
        for i in range(rewards.size(0)):
            if last_step[i]:
                next_return = 0 if done[i] else next_values[i].detach()
            else:
                next_return = next_values[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return
        deltas = returns - values
        # construct the action loss and the value loss
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return value_loss



class MFAC(MF):

    def __init__(self, args, target_net=None):
        super(MFAC, self).__init__(args, target_net)

    def reload_params_to_target(self):
        self.target_net.action_dict.load_state_dict( self.action_dict.state_dict() )
        self.target_net.value_dict.load_state_dict( self.value_dict.state_dict() )

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'layer_1': nn.ModuleList( [ nn.Linear(self.obs_dim, self.hid_dim) for _ in range(self.n_) ] ),\
                                           'layer_2': nn.ModuleList( [ nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_) ] ),\
                                           'action_head': nn.ModuleList( [ nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_) ] )
                                          }
                                        )

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict( {'layer_1': nn.ModuleList( [ nn.Linear(self.obs_dim+self.act_dim, self.hid_dim) for _ in range(self.n_) ] ),\
                                          'layer_2': nn.ModuleList( [ nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_) ] ),\
                                          'value_head': nn.ModuleList( [ nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_) ] )
                                         }
                                       )

    def policy(self, obs, last_act, info={}, stat={}):
        actions = []
        for i in range(self.n_):
            h = torch.relu( self.action_dict['layer_1'][i](obs[:, i, :]) )
            h = torch.relu( self.action_dict['layer_2'][i](h) )
            a = self.action_dict['action_head'][i](h)
            actions.append(a)
        actions = torch.stack(actions, dim=1)
        return actions

    def value(self, obs, act):
        batch_size = obs.size(0)
        values = []
        for i in range(self.n_):
            act_mean = torch.mean(torch.cat( (act[:, :i, :], act[:, (i+1):, :]), dim=1 ), dim=1)
            h = torch.relu( self.value_dict['layer_1'][i]( torch.cat( (obs[:, i, :], act_mean), dim=-1 ) ) )
            h = torch.relu( self.value_dict['layer_2'][i](h) )
            v = self.value_dict['value_head'][i](h)
            values.append(v)
        values = torch.stack(values, dim=1)
        return values

    def get_loss(self, batch):
        batch_size = len(batch.state)
        n = self.args.agent_num
        action_dim = self.args.action_dim
        # collect the transition data
        rewards, last_step, done, actions, last_actions, state, next_state = unpack_data(self.args, batch)
        # construct the computational graph
        action_out = self.policy(state, last_actions)
        values = self.value(state, last_actions)
        values = torch.sum(values*actions, dim=-1)
        values = values.contiguous().view(-1, n)
        if not self.args.target:
            next_action_out = self.policy(next_state, actions)
            next_actions = select_action(self.args, next_action_out, status='train')
            next_values = self.value(next_state, actions)
        else:
            next_action_out = self.target_net.policy(next_state, actions)
            next_actions = select_action(self.args, next_action_out, status='train')
            next_values = self.target_net.value(next_state, next_actions)
            next_values_ = self.target_net.value(next_state, actions)
        next_values_ = torch.sum(torch.softmax(next_values_, dim=-1)*next_values_, dim=-1).view(-1, n)
        next_values = torch.sum(next_values*next_actions, dim=-1)
        next_values = next_values.contiguous().view(-1, n)
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        # calculate the advantages
        assert values.size() == next_values.size()
        assert returns.size() == values.size()
        assert next_values_.size() == returns.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_values_[i].detach()
            else:
                next_return = next_values_[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return
        deltas = returns - values
        advantages = values.detach()
        # construct the action loss and the value loss
        if self.args.continuous:
            action_means = actions.contiguous().view(-1, self.args.action_dim)
            action_stds = cuda_wrapper(torch.ones_like(action_means), self.cuda_)
            log_p_a = normal_log_density(actions.detach(), action_means, action_stds)
            log_prob = log_p_a.clone()
        else:
            log_p_a = action_out
            log_prob = multinomials_log_density(actions, log_p_a).contiguous().view(-1, 1)
        advantages = advantages.contiguous().view(-1, 1)
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        assert log_prob.size() == advantages.size()
        action_loss = -advantages * log_prob
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, log_p_a
