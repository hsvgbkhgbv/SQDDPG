import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from learning_algorithms.reinforce import *
from collections import namedtuple



class IC3Net(Model):

    def __init__(self, args):
        super(IC3Net, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        self.Transition = namedtuple('Transition', ('state', 'action', 'last_action', 'hidden_state', 'last_hidden_state', 'reward', 'next_state', 'done', 'last_step', 'schedule'))

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
        schedules = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.schedule))[0], axis=0), dtype=torch.float), self.cuda_)
        return (rewards, last_step, done, actions, last_actions, hidden_states, last_hidden_states, state, next_state, schedules)

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'encoder': nn.Linear(self.obs_dim, self.hid_dim),\
                                           'g_module_0': nn.Linear(self.hid_dim, self.hid_dim),\
                                           'g_module_1': nn.Linear(self.hid_dim, 2),\
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
        h = torch.relu( self.action_dict['g_module_0'](h) )
        gate = self.action_dict['g_module_1'](h) # shape = (b, n, 2)
        return gate

    def schedule(self, gate):
        return torch.argmin(gate, dim=-1, keepdim=True).float() # act0: comm, act1: not comm

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        batch_size = obs.size(0)
        e = torch.relu(self.action_dict['encoder'](obs))
        h, cell = last_hid[:, :, :self.hid_dim], last_hid[:, :, self.hid_dim:]
        h_ = h.unsqueeze(1).expand(batch_size, self.n_, self.n_, self.hid_dim) # shape = (b, n, h) -> (b, 1, n, h) -> (b, n, n, h)
        mask = self.comm_mask.unsqueeze(0) # shape = (1, n, n)
        mask = mask.expand(batch_size, self.n_, self.n_) # shape = (b, n, n)
        mask = mask.unsqueeze(-1) # shape = (b, n, n, 1)
        mask = mask.expand_as(h_) # shape = (b, n, n, h)
        gate = schedule.unsqueeze(1) # shape = (b, 1, n, 1)
        gate = gate.expand_as(h_) # shape = (b, n, n, h)
        h_ = h_ * gate * mask
        h_ = h_ / (self.n_ - 1)
        c = h_.sum(dim=2) # shape = (b, n, h)
        inp = e + c
        inp = inp.contiguous().view(batch_size*self.n_, self.hid_dim)
        cell = cell.contiguous().view(batch_size*self.n_, self.hid_dim)
        h = h.contiguous().view(batch_size*self.n_, self.hid_dim)
        h, cell = self.action_dict['f_module'](inp, (h, cell))
        cell = cell.contiguous().view(batch_size, self.n_, self.hid_dim)
        h = h.contiguous().view(batch_size, self.n_, self.hid_dim)
        self.lstm_hid = torch.cat([h, cell], dim=-1)
        action = self.action_dict['action_head'](h)
        if batch_size == 1:
            stat['comm_gate'] = schedule.transpose(1, 2).detach().cpu().numpy()
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
        rewards, last_step, done, actions, last_actions, hidden_states, last_hidden_states, state, next_state, schedules = self.unpack_data(batch)
        gate_action_out = self.gate(last_hidden_states[:, :, :self.hid_dim])
        action_out = self.policy(state, schedule=schedules, last_hid=last_hidden_states)
        values = self.value(state).contiguous().view(-1, self.n_)
        next_values = self.value(next_state).contiguous().view(-1, self.n_)
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        assert returns.size() == rewards.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_values[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return
            next_return = returns[i]
        deltas = returns - values
        advantages = deltas.contiguous().view(-1, 1).detach()
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        if self.args.continuous:
            action_means = actions.contiguous().view(-1, self.act_dim)
            action_stds = cuda_wrapper(torch.ones_like(action_means), self.cuda_)
            log_prob_a = normal_log_density(actions.detach(), action_means, action_stds)
        else:
            log_prob_a = multinomials_log_density(actions.detach(), action_out).contiguous().view(-1, 1)
        log_prob_g = torch.log_softmax(gate_action_out, dim=-1).gather(-1, schedules.long()).contiguous().view(-1, 1)
        assert log_prob_a.size() == advantages.size()
        assert log_prob_g.size() == advantages.size()
        action_loss = -advantages * (log_prob_a + log_prob_g)
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, action_out

    def get_episode(self, stat, trainer):
        info = {}
        episode = []
        state = trainer.env.reset()
        action = trainer.init_action
        self.init_hidden(batch_size=1)
        last_hidden_state = self.get_hidden()
        for t in range(self.args.max_steps):
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            action_ = action.clone()
            gate = self.gate(last_hidden_state[:, :, :self.hid_dim])
            schedule = self.schedule(gate)
            action_out = self.policy(state_, schedule=schedule, last_act=action_, last_hid=last_hidden_state, info=info, stat=stat)
            action = select_action(self.args, action_out, status='train', info=info)
            _, actual = translate_action(self.args, action, trainer.env)
            next_state, reward, done, _ = trainer.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            hidden_state = self.get_hidden()
            trans = self.Transition(state,
                                    action.cpu().numpy(),
                                    action_.cpu().numpy(),
                                    hidden_state.cpu().numpy(),
                                    last_hidden_state.cpu().numpy(),
                                    np.array(reward),
                                    next_state,
                                    done,
                                    done_,
                                    schedule.cpu().numpy()
                                   )
            last_hidden_state = hidden_state
            episode.append(trans)
            trainer.steps += 1
            trainer.mean_reward = trainer.mean_reward + 1/trainer.steps*(np.mean(reward) - trainer.mean_reward)
            if done_:
                break
            state = next_state
        stat['mean_reward'] = trainer.mean_reward
        trainer.episodes += 1
        return episode

    def train_process(self, stat, trainer):
        episode = self.get_episode(stat, trainer)
        self.episode_update(trainer, episode, stat)
