import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from learning_algorithms.reinforce import *
from collections import namedtuple



class SchedNet(Model):

    def __init__(self, args, target_net=None):
        super(SchedNet, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'last_step', 'schedule', 'weight'))
        self.eps = 0.5
        self.eps_decay = (self.eps - 0.1) / self.args.train_episodes_num

    def unpack_data(self, batch):
        batch_size = len(batch.state)
        rewards = cuda_wrapper(torch.tensor(batch.reward, dtype=torch.float), self.cuda_)
        last_step = cuda_wrapper(torch.tensor(batch.last_step, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        done = cuda_wrapper(torch.tensor(batch.done, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.action))[0], axis=0), dtype=torch.float), self.cuda_)
        schedules = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.schedule))[0], axis=0), dtype=torch.float), self.cuda_)
        weights = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.weight))[0], axis=0), dtype=torch.float), self.cuda_)
        state = cuda_wrapper(prep_obs(list(zip(batch.state))), self.cuda_)
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda_)
        return (rewards, last_step, done, actions, state, next_state, schedules, weights)

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'message_encoder': nn.ModuleList([nn.Linear(self.obs_dim, self.args.l) for _ in range(self.n_)]),\
                                           'weight_generator_0': nn.ModuleList([nn.Linear(self.obs_dim, self.hid_dim) for _ in range(self.n_)]),\
                                           'weight_generator_1': nn.ModuleList([nn.Linear(self.hid_dim, 1) for _ in range(self.n_)]),\
                                           'action_selector_0': nn.ModuleList([nn.Linear(self.obs_dim+self.args.l*self.args.k, self.hid_dim) for _ in range(self.n_)]),\
                                           'action_selector_1': nn.ModuleList([nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_)])
                                          }
                                        )

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict( {'share_critic': nn.Linear(self.obs_dim*self.n_, self.hid_dim),\
                                          'weight_critic': nn.Linear(self.hid_dim+self.n_, 1),\
                                          'action_critic': nn.Linear(self.hid_dim, 1)
                                         }
                                       )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def weight_generator(self, obs):
        batch_size = obs.size(0)
        w = []
        for i in range(self.n_):
            h = torch.relu( self.action_dict['weight_generator_0'][i](obs[:, i, :]) )
            h = self.action_dict['weight_generator_1'][i](h)
            w.append(h)
        w = torch.stack(w, dim=1).contiguous().view(batch_size, self.n_) # shape = (b, n)
        w = torch.sigmoid(w)
        return w

    def weight_based_scheduler(self, w, exploration):
        if exploration:
            k_ind = cuda_wrapper( torch.randint(low=0, high=w.size(-1), size=(w.size(0), self.args.k)), cuda=self.cuda_ )
        else:
            if self.args.schedule is 'top_k':
                _, k_ind = torch.topk(w, self.args.k, dim=-1, sorted=False)
            elif self.args.schedule is 'softmax_k':
                k_ind = torch.multinomial(torch.softmax(w, dim=-1), self.args.k, replacement=False)
                k_ind, _ = torch.sort(k_ind)
            else:
                raise RuntimeError('Please input the the correct schedule, e.g. top_k or softmax_k.')
        onehot_k_ind = cuda_wrapper(torch.zeros_like(w), cuda=self.cuda_)
        onehot_k_ind.scatter_(-1, k_ind, 1)
        return k_ind, onehot_k_ind

    def message_encoder(self, obs):
        m = []
        for i in range(self.n_):
            h = torch.relu( self.action_dict['message_encoder'][i](obs[:, i, :]) )
            m.append(h)
        m = torch.stack(m, dim=1) # shape = (b, n, h)
        return m

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, gate=None, info={}, stat={}):
        batch_size = obs.size(0)
        m = self.message_encoder(obs)
        c = schedule.unsqueeze(-1).expand(batch_size, self.args.k, self.args.l)
        shared_m = m.gather(1, c.long())
        shared_m = shared_m.unsqueeze(1).expand(batch_size, self.n_, self.args.k, self.args.l) # shape = (b, k, l) -> (b, 1, k, l) -> (b, n, k, l)
        shared_m = shared_m.contiguous().view(batch_size, self.n_, self.args.k*self.args.l) # shape = (b, n, k, l) -> (b, n, k*l)
        action = []
        for i in range(self.n_):
            h = torch.relu( self.action_dict['action_selector_0'][i]( torch.cat([obs[:, i, :], shared_m[:, i, :]], dim=-1) ) )
            h = self.action_dict['action_selector_1'][i](h)
            action.append(h)
        action = torch.stack(action, dim=1)
        return action

    def value(self, obs, w, act=None):
        batch_size = obs.size(0)
        obs = obs.unsqueeze(1).expand(batch_size, self.n_, self.n_, -1) # shape = (b, n, n, o)
        obs = obs.contiguous().view(batch_size, self.n_, -1) # shape = (b, n, n*o)
        w = w.transpose(1, 2).expand(batch_size, self.n_, self.n_) # shape = (b, n, n)
        h = torch.relu( self.value_dict['share_critic'](obs) ) # shape = (b, n, h)
        q = self.value_dict['weight_critic']( torch.cat([h, w], dim=-1) )
        q = q.contiguous().view(batch_size, self.n_)
        v = self.value_dict['action_critic'](h)
        v = v.contiguous().view(batch_size, self.n_)
        return q, v

    def get_loss(self, batch):
        batch_size = len(batch.state)
        rewards, last_step, done, actions, state, next_state, schedules, weights = self.unpack_data(batch)
        action_out = self.policy(state, schedule=schedules)
        weight_action_out = self.weight_generator(state)
        q, v = self.value(state, weights.unsqueeze(-1))
        q_, _ = self.value(state, weight_action_out.unsqueeze(-1))
        next_weight_action_out = self.target_net.weight_generator(next_state)
        next_q, next_v = self.target_net.value(next_state, next_weight_action_out.unsqueeze(-1).detach())
        returns_q = cuda_wrapper(torch.zeros((batch_size, self.n_), dtype=torch.float), self.cuda_)
        returns_v = cuda_wrapper(torch.zeros((batch_size, self.n_), dtype=torch.float), self.cuda_)
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
            returns_q[i] = rewards[i] + self.args.gamma * next_return
        deltas_v = returns_v - v
        deltas_q = returns_q - q
        advantages_v = deltas_v.contiguous().view(-1, 1).detach()
        advantages_q = q_.contiguous().view(-1, 1)
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        if self.args.continuous:
            action_means = actions.contiguous().view(-1, self.act_dim)
            action_stds = cuda_wrapper(torch.ones_like(action_means), self.cuda_)
            log_prob_a = normal_log_density(actions, action_means, action_stds)
        else:
            log_prob_a = multinomials_log_density(actions, action_out).contiguous().view(-1, 1)
        assert log_prob_a.size() == advantages_v.size()
        action_loss = - advantages_v*log_prob_a - advantages_q
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas_v.pow(2).view(-1).mean() + deltas_q.pow(2).view(-1).mean()
        return action_loss, value_loss, action_out

    def train_process(self, stat, trainer):
        info = {}
        state = trainer.env.reset()
        for t in range(self.args.max_steps):
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            weight = self.weight_generator(state_).detach()
            epsilon = np.random.rand()
            if epsilon < self.eps:
                schedule, onehot_schedule = self.weight_based_scheduler(weight, exploration=True)
            else:
                schedule, onehot_schedule = self.weight_based_scheduler(weight, exploration=False)
            stat['schedule'] = onehot_schedule.unsqueeze(1).cpu().numpy()
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            epsilon = np.random.rand()
            if epsilon < self.eps:
                action_out = cuda_wrapper( torch.rand((1, self.n_, self.act_dim)), cuda=self.cuda_ )
            else:
                action_out = self.policy(state_, schedule=schedule, info=info, stat=stat)
            action = select_action(self.args, action_out, status='train', exploration=True, info=info)
            _, actual = translate_action(self.args, action, trainer.env)
            next_state, reward, done, _ = trainer.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            trans = self.Transition(state,
                                    action.cpu().numpy(),
                                    np.array(reward),
                                    next_state,
                                    done,
                                    done_,
                                    schedule.cpu().numpy(),
                                    weight.cpu().numpy()
                                   )
            self.transition_update(trainer, trans, stat)
            trainer.steps += 1
            trainer.mean_reward = trainer.mean_reward + 1/trainer.steps*(np.mean(reward) - trainer.mean_reward)
            stat['mean_reward'] = trainer.mean_reward
            if done_:
                break
            state = next_state
        self.eps -= self.eps_decay
        trainer.episodes += 1
