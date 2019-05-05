import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from learning_algorithms.reinforce import *



class SchedNet(Model):

    def __init__(self, args, target_net=None):
        super(SchedNet, self).__init__(args)
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

    def unpack_data(self, batch):
        batch_size = len(batch.state)
        rewards = cuda_wrapper(torch.tensor(batch.reward, dtype=torch.float), self.cuda_)
        last_step = cuda_wrapper(torch.tensor(batch.last_step, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        done = cuda_wrapper(torch.tensor(batch.done, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.action))[0], axis=0), dtype=torch.float), self.cuda_)
        last_actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.last_action))[0], axis=0), dtype=torch.float), self.cuda_)
        schedules = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.schedule))[0], axis=0), dtype=torch.float), self.cuda_)
        state = cuda_wrapper(prep_obs(list(zip(batch.state))), self.cuda_)
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda_)
        return (rewards, last_step, done, actions, last_actions, state, next_state, schedules)

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'message_encoder': nn.ModuleList([nn.Linear(self.obs_dim, self.hid_dim) for _ in range(self.n_)]),\
                                           'weight_generator': nn.ModuleList([nn.Linear(self.obs_dim, 1) for _ in range(self.n_)]),\
                                           'action_selector': nn.ModuleList([nn.Linear(self.obs_dim+self.hid_dim*self.n_, self.act_dim) for _ in range(self.n_)])
                                          }
                                        )

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict( {'share_critic': nn.Linear(self.obs_dim, self.hid_dim),\
                                          'weight_critic': nn.Linear(self.hid_dim+1, 1),\
                                          'action_critic': nn.Linear(self.hid_dim, 1)
                                         }
                                       )

    def construct_model(self):
        self.comm_mask = cuda_wrapper(torch.ones(self.n_, self.n_) - torch.eye(self.n_, self.n_), self.cuda_)
        self.construct_value_net()
        self.construct_policy_net()

    def weight_generator(self, obs):
        batch_size = obs.size(0)
        w = []
        for i in range(self.n_):
            w.append(self.action_dict['weight_generator'][i](obs[:, i, :]))
        self.w = torch.stack(w, dim=1).contiguous().view(batch_size, self.n_) # shape = (batch_size, n)
        return self.w

    def weight_based_scheduler(self, w):
        if self.args.schedule is 'top_k':
            _, k_ind = torch.topk(w, self.args.k, dim=-1, sorted=False)
        elif self.args.schedule is 'softmax_k':
            k_ind = torch.multinomial(w, self.args.k)
            k_ind, _ = torch.sort(k_ind)
        else:
            raise RuntimeError('Please input the the correct schedule, e.g. top_k or softmax_k.')
        onehot_k_ind = cuda_wrapper(torch.zeros_like(w), cuda=self.cuda_)
        onehot_k_ind.scatter_(-1, k_ind, 1)
        return onehot_k_ind

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, gate=None, info={}, stat={}):
        batch_size = obs.size(0)
        m = []
        for i in range(self.n_):
            m.append(self.action_dict['message_encoder'][i](obs[:, i, :]))
        m = torch.stack(m, dim=1) # shape = (batch_size, n, hid_size)
        m = m.unsqueeze(1).expand(batch_size, self.n_, self.n_, self.hid_dim) # shape = (batch_size, n, hid_size) -> (batch_size, 1, n, hid_size) -> (batch_size, n, n, hid_size)
        c = schedule # shape = (batch_size, n)
        c = c.unsqueeze(1).expand(batch_size, self.n_, self.n_) # shape = (batch_size, n) -> (batch_size, 1, n) -> (batch_size, n, n)
        c = c.unsqueeze(-1).expand(batch_size, self.n_, self.n_, self.hid_dim) # shape = (batch_size, n, n) -> (batch_size, n, n, 1) -> (batch_size, n, n, hid_size)
        shared_m = m * c.float()
        shared_m = shared_m.contiguous().view(batch_size, self.n_, self.hid_dim*self.n_)
        action = []
        for i in range(self.n_):
            action.append( self.action_dict['action_selector'][i]( torch.cat([obs[:, i, :], shared_m[:, i, :]], dim=-1) ) )
        action = torch.stack(action, dim=1)
        if batch_size == 1:
            stat['schedule'] = schedule.unsqueeze(1).detach().cpu().numpy()
        return action

    def value(self, obs, w, act=None):
        shared_param = self.value_dict['share_critic'](obs)
        q = self.value_dict['weight_critic']( torch.cat([shared_param, w], dim=-1) )
        v = self.value_dict['action_critic'](shared_param)
        return q.contiguous().view(-1, self.n_), v.contiguous().view(-1, self.n_)

    def get_loss(self, batch):
        batch_size = len(batch.state)
        n = self.args.agent_num
        # collect the transition data
        rewards, last_step, done, actions, last_actions, state, next_state, schedules = self.unpack_data(batch)
        # construct the computational graph
        action_out = self.policy(state, schedule=schedules)
        weight_action_out = self.weight_generator(state)
        q, v = self.value(state, weight_action_out.unsqueeze(-1).detach())
        q_, _ = self.value(state, weight_action_out.unsqueeze(-1))
        # get the next actions and the next values
        next_q, next_v = self.target_net.value(next_state, self.target_net.weight_generator(next_state).unsqueeze(-1).detach())
        returns_q = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        returns_v = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        # calculate the return
        assert returns_v.size() == rewards.size()
        assert returns_q.size() == rewards.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_v[i].detach()
            else:
                next_return = next_v[i].detach()
            returns_v[i] = rewards[i] + self.args.gamma * next_return
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_q[i].detach()
            else:
                next_return = next_q[i].detach()
            returns_v[i] = rewards[i] + self.args.gamma * next_return
        # construct the action loss and the value loss
        deltas_v = returns_v - v
        deltas_q = returns_q - q
        advantages_v = deltas_v.contiguous().view(-1, 1).detach()
        advantages_q = q_.contiguous().view(-1, 1)
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
        # log_prob_g = weight_action_out.gather(-1, schedules.long()).contiguous().view(-1, 1)
        assert log_prob_a.size() == advantages_v.size()
        action_loss = - advantages_v * log_prob_a - advantages_q
        action_loss = action_loss.sum() / batch_size
        value_loss = ( deltas_v.pow(2).view(-1).sum() + deltas_q.pow(2).view(-1).sum() ) / batch_size
        return action_loss, value_loss, log_p_a
