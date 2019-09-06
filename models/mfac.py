import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from collections import namedtuple



class MFAC(Model):

    def __init__(self, args, target_net=None):
        super(MFAC, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        if self.args.epsilon_softmax:
            self.eps_delta = (args.softmax_eps_init - args.softmax_eps_end) / args.train_episodes_num
            self.eps = args.softmax_eps_init
        self.Transition = namedtuple('Transition', ('state', 'action','reward', 'next_state', 'done', 'last_step'))

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
        l1 = nn.Linear(self.obs_dim, self.hid_dim)
        l2 = nn.Linear(self.hid_dim, self.hid_dim)
        a = nn.Linear(self.hid_dim, self.act_dim)
        if self.args.shared_parameters:
            self.action_dict = nn.ModuleDict( {'layer_1': nn.ModuleList([ l1 for _ in range(self.n_)]),\
                                               'layer_2': nn.ModuleList([ l2 for _ in range(self.n_)]),\
                                               'action_head': nn.ModuleList([ a for _ in range(self.n_)])
                                              }
                                            )
        else:
            self.action_dict = nn.ModuleDict( {'layer_1': nn.ModuleList([nn.Linear(self.obs_dim, self.hid_dim) for _ in range(self.n_)]),\
                                               'layer_2': nn.ModuleList([nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_)]),\
                                               'action_head': nn.ModuleList([nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_)])
                                              }
                                            )

    def construct_value_net(self):
        if self.args.shared_parameters:
            l1 = nn.Linear( self.obs_dim+self.act_dim, self.hid_dim )
            l2 = nn.Linear(self.hid_dim, self.hid_dim)
            v = nn.Linear(self.hid_dim, 1)
            self.value_dict = nn.ModuleDict( {'layer_1': nn.ModuleList([ l1 for _ in range(self.n_)]),\
                                              'layer_2': nn.ModuleList([ l2 for _ in range(self.n_)]),\
                                              'value_head': nn.ModuleList([ v for _ in range(self.n_)])
                                             }
                                             )
        else:
            self.value_dict = nn.ModuleDict( {'layer_1': nn.ModuleList([nn.Linear(self.obs_dim+self.act_dim, self.hid_dim ) for _ in range(self.n_)]),\
                                              'layer_2': nn.ModuleList([nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_)]),\
                                              'value_head': nn.ModuleList([nn.Linear(self.hid_dim, 1) for _ in range(self.n_)])
                                             }
                                             )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
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
        act = act.unsqueeze(1).expand(batch_size, self.n_, self.n_, self.act_dim) # shape = (b, n, a) -> (b, 1, n, a) -> (b, n, n, a)
        comm_mask = self.comm_mask.unsqueeze(0).unsqueeze(-1).expand_as(act)
        act = act * comm_mask
        act = act.contiguous().view(batch_size, self.n_, -1) # shape = (b, n, a*n)
        obs = obs.unsqueeze(1).expand(batch_size, self.n_, self.n_, self.obs_dim).contiguous().view(batch_size, self.n_, -1) # shape = (b, n, o) -> (b, 1, n, o) -> (b, n, n, o)
        inp = torch.cat((obs, act), dim=-1) # shape = (b, n, o*n+a*n)
        
    def value(self, obs, act):
        mean_act = torch.mean(act, dim=1, keepdim=True).repeat(1, self.n_,  1) # shape = (b, n, a) -> (b, 1, a) -> (b, n, a)
        inp = torch.cat((obs, mean_act),dim=-1) # shape = (b, n, o+a) 
        values = []
        for i in range(self.n_):
            h = torch.relu( self.value_dict['layer_1'][i](inp[:, i, :]) )
            h = torch.relu( self.value_dict['layer_2'][i](h) )
            v = self.value_dict['value_head'][i](h)
            values.append(v)
        values = torch.stack(values, dim=1)
        return values

    def get_loss(self, batch):
        batch_size = len(batch.state)
        rewards, last_step, done, actions, state, next_state = self.unpack_data(batch)
        # calculate values
        action_out = self.policy(state)
        values = self.value(state, actions).contiguous().view(-1, self.n_)
        # calculate next values
        next_action_out = self.policy(next_state)
        next_actions = select_action(self.args, next_action_out.detach(), status='train')
        next_values = self.value(next_state, next_actions).contiguous().view(-1, self.n_)
        
        # calculate the return
        returns = cuda_wrapper(torch.zeros((batch_size, self.n_), dtype=torch.float), self.cuda_)
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
            log_prob_a = normal_log_density(actions.detach(), action_means, action_stds)
        else:
            log_prob_a = multinomials_log_density(actions.detach(), action_out).contiguous().view(-1,1)
        assert log_prob_a.size() == advantages.size()
        action_loss = -advantages * log_prob_a
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, action_out
        
    def get_episode(self, stat, trainer):
        episode = []
        info = {}
        state = trainer.env.reset()
        if self.args.reward_record_type is 'episode_mean_step':
            trainer.mean_reward = 0
            trainer.mean_success = 0

        for t in range(self.args.max_steps):
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            action_out = self.policy(state_)
            action = select_action(self.args, torch.log_softmax(action_out, dim=-1), status='train', info=info)
            _, actual = translate_action(self.args, action, trainer.env)
            next_state, reward, done, debug = trainer.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            trans = self.Transition(state,
                        action.cpu().numpy(),
                        np.array(reward),
                        next_state,
                        done,
                        done_
                       )
            episode.append(trans)
            success = debug['success'] if 'success' in debug else 0.0
            trainer.steps += 1
            if self.args.reward_record_type is 'mean_step':
                trainer.mean_reward = trainer.mean_reward + 1/trainer.steps*(np.mean(reward) - trainer.mean_reward)
                trainer.mean_success = trainer.mean_success + 1/trainer.steps*(success - trainer.mean_success)
            elif self.args.reward_record_type is 'episode_mean_step':
                trainer.mean_reward = trainer.mean_reward + 1/(t+1)*(np.mean(reward) - trainer.mean_reward)
                trainer.mean_success = trainer.mean_success + 1/(t+1)*(success - trainer.mean_success)
            else:
                raise RuntimeError('Please enter a correct reward record type, e.g. mean_step or episode_mean_step.')
            if done_:
                break
            state = next_state
        stat['turn'] = t+1
        stat['mean_reward'] = trainer.mean_reward
        stat['mean_success'] = trainer.mean_success
        trainer.episodes += 1
        return episode

    def train_process(self, stat, trainer):
        episode = self.get_episode(stat, trainer)
        self.episode_update(trainer, episode, stat)
