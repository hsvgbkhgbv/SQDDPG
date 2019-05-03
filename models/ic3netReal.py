import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from learning_algorithms.reinforce import *



class IC3Net(Model):

    def __init__(self, args):
        super(IC3Net, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)

    def unpack_data(self, batch):
        batch_size = len(batch.state)
        rewards = cuda_wrapper(torch.tensor(batch.reward, dtype=torch.float), self.cuda_)
        last_step = cuda_wrapper(torch.tensor(batch.last_step, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        done = cuda_wrapper(torch.tensor(batch.done, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.action))[0], axis=0), dtype=torch.float), self.cuda_)
        last_actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.last_action))[0], axis=0), dtype=torch.float), self.cuda_)
        hidden_states = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.hidden_state))[0], axis=0), dtype=torch.float), self.cuda_)
        last_hidden_states = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.last_hidden_state))[0], axis=0), dtype=torch.float), self.cuda_)
        state = cuda_wrapper(prep_obs(list(zip(batch.state))), self.cuda_)
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda_)
        return (rewards, last_step, done, actions, last_actions, hidden_states, last_hidden_states, state, next_state)

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'encoder': nn.Linear(self.obs_dim, self.hid_dim),\
                                           'g_module': nn.Linear(self.hid_dim, 2),\
                                           'f_module': nn.LSTMCell(self.hid_dim, self.hid_dim),\
                                           'action_head': nn.Linear(self.hid_dim, self.act_dim)
                                          }
                                        )

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict()
        self.value_dict['value_body'] = nn.Linear(self.obs_dim, self.hid_dim)
        self.value_dict['value_head'] = nn.Linear(self.hid_dim, 1)

    def construct_model(self):
        self.comm_mask = cuda_wrapper(torch.ones(self.n_, self.n_) - torch.eye(self.n_, self.n_), self.cuda_)
        self.construct_value_net()
        self.construct_policy_net()

    def gate(self, h):
        gate = self.action_dict['g_module'](h) # shape = (batch_size, n, 2)
        return gate

    def policy(self, obs, last_act=None, last_hid=None, info={}, stat={}):
        batch_size = obs.size(0)
        # encode observation
        e = torch.relu(self.action_dict['encoder'](obs))
        # get the initial state
        # if info.get('start', False):
        #     h, cell = self.init_hidden(batch_size)
        h, cell = last_hid[:, :, :self.hid_dim], last_hid[:, :, self.hid_dim:]
        # get the agent mask
        # num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        # conduct the main process of communication
        # h_ = h.contiguous().view(batch_size, self.n_, self.hid_dim)
        # define the gate function
        gate_ = self.gate(h).detach()
        gate_ = torch.argmin(gate_, dim=-1, keepdim=True).float() # act0: comm, act1: not comm
        # shape = (batch_size, n, hid_size)->(batch_size, 1, n, hid_size)->(batch_size, n, n, hid_size)
        h_ = h.unsqueeze(1).expand(batch_size, self.n_, self.n_, self.hid_dim)
        # construct the communication mask
        mask = self.comm_mask.unsqueeze(0) # shape = (1, n, n)
        mask = mask.expand(batch_size, self.n_, self.n_) # shape = (batch_size, n, n)
        mask = mask.unsqueeze(-1) # shape = (batch_size, n, n, 1)
        mask = mask.expand_as(h_) # shape = (batch_size, n, n, hid_size)
        # construct the commnication gate
        gate = gate_.unsqueeze(1) # shape = (batch_size, 1, n, 1)
        gate = gate.expand_as(h_) # shape = (batch_size, n, n, hid_size)
        # mask each agent itself (collect the hidden state of other agents)
        h_ = h_ * gate * mask
        # mask the dead agent
        # h_ = h_ * agent_mask * agent_mask.transpose(1, 2)
        # average the hidden state
        # if num_agents_alive > 1: h_ = h_ / (num_agents_alive - 1)
        h_ = h_ / (self.n_ - 1)
        # calculate the communication vector
        c = h_.sum(dim=2) # shape = (batch_size, n, hid_size)
        inp = e + c
        inp = inp.contiguous().view(batch_size*self.n_, self.hid_dim)
        # f_moudle
        cell = cell.contiguous().view(batch_size*self.n_, self.hid_dim)
        h = h.contiguous().view(batch_size*self.n_, self.hid_dim)
        h, cell = self.action_dict['f_module'](inp, (h, cell))
        cell = cell.contiguous().view(batch_size, self.n_, self.hid_dim)
        h = h.contiguous().view(batch_size, self.n_, self.hid_dim)
        self.lstm_hid = torch.cat([h, cell], dim=-1)
        # calculate the action vector (policy)
        action = self.action_dict['action_head'](h)
        if batch_size == 1:
            stat['comm_gate'] = gate_.transpose(1, 2).detach().cpu().numpy()
        return action

    def value(self, obs, act=None):
        h = self.value_dict['value_body'](obs)
        h = torch.relu(h)
        v = self.value_dict['value_head'](h)
        return v

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        self.lstm_hid = cuda_wrapper(torch.zeros(batch_size, self.n_, self.hid_dim*2), self.cuda_)

    def get_hidden(self):
        return self.lstm_hid.detach()

    def get_loss(self, batch):
        batch_size = len(batch.state)
        n = self.args.agent_num
        # collect the transition data
        rewards, last_step, done, actions, last_actions, hidden_states, last_hidden_states, state, next_state = self.unpack_data(batch)
        # construct the computational graph
        action_out = self.policy(state, last_hid=last_hidden_states)
        gate_action_out = self.gate(last_hidden_states[:, :, :self.hid_dim])
        values = self.value(state).contiguous().view(-1, n)
        # get the next actions and the next values
        next_values = self.value(next_state).contiguous().view(-1, n)
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        # calculate the return
        assert returns.size() == rewards.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_values[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return
            next_return = returns[i]
        # construct the action loss and the value loss
        deltas = returns - values
        advantages = deltas.contiguous().view(-1, 1).detach()
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        if self.args.continuous:
            action_means = actions.contiguous().view(-1, self.args.action_dim)
            action_stds = cuda_wrapper(torch.ones_like(action_means), self.cuda_)
            log_p_a = normal_log_density(actions.detach(), action_means, action_stds)
            log_prob_a = log_p_a.clone()
        else:
            log_p_a = action_out
            log_prob_a = multinomials_log_density(actions.detach(), log_p_a).contiguous().view(-1, 1)
        log_prob_g = gate_action_out.gather(-1, torch.argmax(gate_action_out, dim=-1, keepdim=True).detach()).contiguous().view(-1, 1)
        assert log_prob_a.size() == advantages.size()
        assert log_prob_g.size() == advantages.size()
        action_loss = -advantages * (log_prob_a + log_prob_g)
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, log_p_a
