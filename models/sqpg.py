# just copy COMA to make program run
import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from collections import namedtuple



class SQPG(Model):

    def __init__(self, args, target_net=None):
        super(SQPG, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.sample_size = self.args.sample_size
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'last_step'))

    def update_eps(self):
        self.eps -= self.eps_delta

    def unpack_data(self, batch):
        batch_size = len(batch.state)
        rewards = cuda_wrapper(torch.tensor(batch.reward, dtype=torch.float), self.cuda_)
        last_step = cuda_wrapper(torch.tensor(batch.last_step, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        done = cuda_wrapper(torch.tensor(batch.done, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.action))[0], axis=0), dtype=torch.float), self.cuda_)
        state = cuda_wrapper(prep_obs(list(zip(batch.state))), self.cuda_)
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda_)
        return (rewards, last_step, done, actions, state, next_state)

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'encoder': nn.ModuleList([nn.Linear(self.obs_dim, self.hid_dim) for _ in range(self.n_)]),\
                                           'linear_layer': nn.ModuleList([nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_)]),\
                                           'action_head': nn.ModuleList([nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_)])
                                          }
                                        )

    def construct_value_net(self):
        # self.value_dict = nn.ModuleDict( {'layer_1': nn.ModuleList([nn.Linear( self.obs_dim*self.n_+self.act_dim*self.n_, self.hid_dim ) for _ in range(self.n_)]),\
        #                                   'layer_2': nn.ModuleList([nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_)]),\
        #                                   'value_head': nn.ModuleList([nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_)])
        #                                  }
        #                                  )
        l1 = nn.Linear( self.obs_dim*self.n_+self.act_dim*self.n_, self.hid_dim )
        l2 = nn.Linear(self.hid_dim, self.hid_dim)
        l3 = nn.Linear(self.hid_dim, self.act_dim)
        self.value_dict = nn.ModuleDict( {'layer_1': nn.ModuleList([l1 for _ in range(self.n_)]),\
                                          'layer_2': nn.ModuleList([l2 for _ in range(self.n_)]),\
                                          'value_head': nn.ModuleList([l3 for _ in range(self.n_)])
                                         }
                                         )

    def construct_model(self):
        self.comm_mask = cuda_wrapper(torch.ones(self.n_, self.n_) - torch.eye(self.n_, self.n_), self.cuda_)
        self.construct_value_net()
        self.construct_policy_net()

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        batch_size = obs.size(0)
        actions = []
        for i in range(self.n_):
            enc = torch.relu( self.action_dict['encoder'][i](obs[:, i, :]) )
            h = torch.relu( self.action_dict['linear_layer'][i](enc) )
            a = self.action_dict['action_head'][i](h)
            actions.append(a)
        actions = torch.stack(actions, dim=1)
        return actions

    def value(self, obs, act):
        batch_size = obs.size(0)
        act = act.unsqueeze(1).expand(batch_size, self.n_, self.n_, self.act_dim) # shape = (b, n, a) -> (b, 1, n, a) -> (b, n, n, a)
        act = act * self.comm_mask.unsqueeze(0).unsqueeze(-1).expand_as(act) # shape = (n, n) -> (1, n, n) -> (1, n, n, 1) -> (b, n, n, a)
        act = act.contiguous().view(batch_size, self.n_, -1) # shape = (b, n, a*n)
        obs = obs.unsqueeze(1).expand(batch_size, self.n_, self.n_, self.obs_dim).contiguous().view(batch_size, self.n_, -1)
        inp = torch.cat((obs, act), dim=-1) # shape = (b, n, o*n+a*n)
        values = []
        for i in range(self.n_):
            h = torch.relu( self.value_dict['layer_1'][i](inp[:, i, :]) )
            h = torch.relu( self.value_dict['layer_2'][i](h) )
            v = self.value_dict['value_head'][i](h)
            values.append(v)
        values = torch.stack(values, dim=1)
        return values

    def sample_coalitions(self, obs):
        batch_size = obs.size(0)
        grand_coalitions = cuda_wrapper( torch.multinomial(torch.ones(batch_size*self.sample_size, self.n_)/self.n_, self.n_, replacement=False), self.cuda_ )
        grand_coalitions = grand_coalitions.contiguous().view(batch_size, self.sample_size, self.n_) # shape = (b, n_s, n)
        grand_coalitions = grand_coalitions.unsqueeze(2).expand(batch_size, self.sample_size, self.n_, self.n_) # shape = (b, n_s, n) -> (b, n_s, 1, n) -> (b, n_s, n, n)
        coalition_map = cuda_wrapper( torch.ones_like(grand_coalitions), self.cuda_ ) # shape = (b, n_s, n, n)
        for b in range(batch_size):
            for s in range(self.sample_size):
                for i in range(self.n_):
                    agent_index = (grand_coalitions[b, s, i, :] == i).nonzero()
                    coalition_map[b, s, i, agent_index:] = 0
        return coalition_map, grand_coalitions

    def grand_coalition_value(self, obs, act):
        batch_size = obs.size(0)
        _, grand_coalitions = self.sample_coalitions(obs) # shape = (b, n_s, n, n)
        coalition_map = 1 - (torch.arange(self.n_).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand_as(grand_coalitions) == grand_coalitions).float()
        grand_coalitions = grand_coalitions.unsqueeze(-1).expand(batch_size, self.sample_size, self.n_, self.n_, self.act_dim) # shape = (b, n_s, n, n, a)
        act = act.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_, self.n_, self.act_dim).gather(3, grand_coalitions) # shape = (b, n, a) -> (b, 1, 1, n, a) -> (b, n_s, n, n, a)
        act_map = coalition_map.unsqueeze(-1).float() # shape = (b, n_s, n, n, 1)
        act = act * act_map
        act = act.contiguous().view(batch_size, self.sample_size, self.n_, -1) # shape = (b, n_s, n, n*a)
        obs = obs.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_, self.n_, self.obs_dim) # shape = (b, n, o) -> (b, 1, n, o) -> (b, 1, 1, n, o) -> (b, n_s, n, n, o)
        obs = obs.contiguous().view(batch_size, self.sample_size, self.n_, self.n_*self.obs_dim) # shape = (b, n_s, n, n, o) -> (b, n_s, n, n*o)
        inp = torch.cat((obs, act), dim=-1)
        values = []
        for i in range(self.n_):
            h = torch.relu( self.value_dict['layer_1'][i](inp[:, :, i, :]) )
            h = torch.relu( self.value_dict['layer_2'][i](h) )
            v = self.value_dict['value_head'][i](h)
            values.append(v)
        values = torch.stack(values, dim=2)
        return values

    def small_coalition_value(self, obs, act):
        batch_size = obs.size(0)
        coalition_map, grand_coalitions = self.sample_coalitions(obs) # shape = (b, n_s, n, n)
        grand_coalitions = grand_coalitions.unsqueeze(-1).expand(batch_size, self.sample_size, self.n_, self.n_, self.act_dim) # shape = (b, n_s, n, n, a)
        act = act.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_, self.n_, self.act_dim).gather(3, grand_coalitions) # shape = (b, n, a) -> (b, 1, 1, n, a) -> (b, n_s, n, n, a)
        act_map = coalition_map.unsqueeze(-1).float() # shape = (b, n_s, n, n, 1)
        act = act * act_map
        act = act.contiguous().view(batch_size, self.sample_size, self.n_, -1) # shape = (b, n_s, n, n*a)
        obs = obs.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_, self.n_, self.obs_dim) # shape = (b, n, o) -> (b, 1, n, o) -> (b, 1, 1, n, o) -> (b, n_s, n, n, o)
        obs = obs.contiguous().view(batch_size, self.sample_size, self.n_, self.n_*self.obs_dim) # shape = (b, n_s, n, n, o) -> (b, n_s, n, n*o)
        inp = torch.cat((obs, act), dim=-1)
        values = []
        for i in range(self.n_):
            h = torch.relu( self.value_dict['layer_1'][i](inp[:, :, i, :]) )
            h = torch.relu( self.value_dict['layer_2'][i](h) )
            v = self.value_dict['value_head'][i](h)
            values.append(v)
        values = torch.stack(values, dim=2)
        return values

    def get_loss(self, batch):
        info = {}
        batch_size = len(batch.state)
        rewards, last_step, done, actions, state, next_state = self.unpack_data(batch)
        action_out = self.policy(state, info=info)
        # values_ = self.value(state, actions)
        values_ = self.grand_coalition_value(state, actions).mean(dim=1) # shape = (b, n, a)
        if self.args.q_func:
            values = torch.sum(values_*actions, dim=-1)
        values = values.contiguous().view(-1, self.n_)
        next_action_out = self.target_net.policy(next_state, info=info)
        # next_action_out = self.policy(next_state, last_act=actions, last_hid=hidden_state, info=info)
        next_actions = select_action(self.args, next_action_out, status='train', info=info, exploration=False)
        # next_actions_ = next_actions.unsqueeze(1)
        # next_values_ = self.target_net.value(next_state, next_actions)
        next_values_ = self.target_net.grand_coalition_value(next_state, next_actions).mean(dim=1) # shape = (b, n, a)
        # next_coalition_values_ = self.target_net.coalition_value(next_state, next_actions)
        # next_values = self.value(next_state, next_actions)
        coalition_values_ = self.small_coalition_value(state, actions) # shape = (b, n_s, n, a)
        actions_ = actions.unsqueeze(1)
        coalition_values = torch.sum(coalition_values_*actions_, dim=-1) # shape = (b, n_s, n, 1)
        if self.args.q_func:
            next_values = torch.sum(next_values_*next_actions, dim=-1)
            # next_coalition_values = torch.sum(next_coalition_values_*next_actions_, dim=-1) # shape = (b, n_s, n)
        # next_values = ( next_coalition_values - torch.sum(next_coalition_values_*torch.softmax(next_action_out.unsqueeze(1).expand_as(next_coalition_values_), dim=-1), dim=-1) ).mean(dim=1).detach() # shape = (b, n)
        # next_values = next_values.sum(dim=1, keepdim=True).expand(batch_size, self.n_)
        # next_values = next_values.contiguous().view(-1, self.n_)
        assert values.size() == next_values.size()
        returns = cuda_wrapper(torch.zeros((batch_size, self.n_), dtype=torch.float), self.cuda_)
        assert returns.size() == rewards.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_values[i].detach()
            else:
                next_return = next_values[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return
        deltas = returns - values
        shapley_q = ( coalition_values - torch.sum(coalition_values_*torch.softmax(action_out.unsqueeze(1).expand_as(coalition_values_), dim=-1), dim=-1) ).mean(dim=1).detach()
        log_prob = multinomials_log_density(actions, action_out).contiguous().view(-1, 1)
        advantages = shapley_q.contiguous().view(-1, 1)
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        assert log_prob.size() == advantages.size()
        action_loss = - advantages * log_prob
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, action_out

    def train_process(self, stat, trainer):
        info = {}
        state = trainer.env.reset()
        if self.args.reward_record_type is 'episode_mean_step':
            trainer.mean_reward = 0
        for t in range(self.args.max_steps):
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            action_out = self.policy(state_, info=info, stat=stat)
            action = select_action(self.args, action_out, status='train', info=info, exploration=True)
            _, actual = translate_action(self.args, action, trainer.env)
            next_state, reward, done, _ = trainer.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            trans = self.Transition(state,
                                    action.cpu().numpy(),
                                    np.array(reward),
                                    next_state,
                                    done,
                                    done_
                                   )
            self.transition_update(trainer, trans, stat)
            trainer.steps += 1
            if self.args.reward_record_type is 'mean_step':
                trainer.mean_reward = trainer.mean_reward + 1/trainer.steps*(np.mean(reward) - trainer.mean_reward)
            elif self.args.reward_record_type is 'episode_mean_step':
                trainer.mean_reward = trainer.mean_reward + 1/(t+1)*(np.mean(reward) - trainer.mean_reward)
            else:
                raise RuntimeError('Please enter a correct reward record type, e.g. mean_step or episode_mean_step.')
            stat['mean_reward'] = trainer.mean_reward
            if done_:
                break
            state = next_state
        trainer.episodes += 1
