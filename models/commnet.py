import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from learning_algorithms.reinforce import *
from collections import namedtuple



class CommNet(Model):

    def __init__(self, args):
        super(CommNet, self).__init__(args)
        self.comm_iters = self.args.comm_iters
        self.rl = REINFORCE(self.args)
        self.identifier()
        self.construct_model()
        self.apply(self.init_weights)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'last_step'))

    def identifier(self):
        if self.comm_iters == 0:
            raise RuntimeError('Please guarantee the comm iters is at least greater equal to 1.')
        elif self.comm_iters < 2:
            raise RuntimeError('Please use IndependentCommNet if the comm iters is set to 1.')

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
        self.action_dict = nn.ModuleDict( {'encoder': nn.Linear(self.obs_dim, self.hid_dim),\
                                           'f_module': nn.Linear(self.hid_dim, self.hid_dim),\
                                           'c_module': nn.Linear(self.hid_dim, self.hid_dim),\
                                           'action_head': nn.Linear(self.hid_dim, self.act_dim)
                                          }
                                        )
        self.action_dict['f_modules'] = nn.ModuleList( [ self.action_dict['f_module'] for _ in range(self.comm_iters) ] )
        self.action_dict['c_modules'] = nn.ModuleList( [ self.action_dict['c_module'] for _ in range(self.comm_iters) ] )
        if self.args.skip_connection:
            self.action_dict['e_module'] = nn.Linear(self.hid_dim, self.hid_dim)
            self.action_dict['e_modules'] = nn.ModuleList( [ self.action_dict['e_module'] for _ in range(self.comm_iters) ] )

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict()
        self.value_dict['value_body'] = nn.Linear(self.obs_dim, self.hid_dim)
        self.value_dict['value_head'] = nn.Linear(self.hid_dim, 1)

    def construct_model(self):
        self.comm_mask = cuda_wrapper(torch.ones(self.n_, self.n_) - torch.eye(self.n_, self.n_), self.cuda_)
        self.construct_value_net()
        self.construct_policy_net()

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        batch_size = obs.size(0)
        if self.args.skip_connection:
            e = torch.relu( self.action_dict['encoder'](obs) )
            h = torch.zeros_like(e)
        else:
            h = torch.tanh( self.action_dict['encoder'](obs) )
        for i in range(self.comm_iters):
            h_ = h.unsqueeze(1).expand(batch_size, self.n_, self.n_, self.hid_dim) # shape = (b, n, h)->(b, 1, n, h)->(b, n, n, h)
            mask = self.comm_mask.unsqueeze(0) # shape = (1, n, n)
            mask = mask.expand(batch_size, self.n_, self.n_) # shape = (b, n, n)
            mask = mask.unsqueeze(-1) # shape = (b, n, n, 1)
            mask = mask.expand_as(h_) # shape = (b, n, n, h)
            h_ = h_ * mask
            h_ = h_ / (self.n_ - 1)
            c = h_.sum(dim=2) if i != 0 else torch.zeros_like(h) # shape = (b, n, h)
            if self.args.skip_connection:
                # h_{j}^{i+1} = \sigma(H_j * h_j^{i+1} + C_j * c_j^{i+1} + E_{j} * e_j^{i+1})
                h = torch.tanh( sum( [ self.action_dict['f_modules'][i](h), self.action_dict['c_modules'][i](c), self.action_dict['e_modules'][i](e) ] ) )
            else:
                # h_{j}^{i+1} = \sigma(H_j * h_j^{i+1} + C_j * c_j^{i+1})
                h = torch.tanh( sum( [ self.action_dict['f_modules'][i](h), self.action_dict['c_modules'][i](c) ] ) )
        action = self.action_dict['action_head'](h)
        return action

    def value(self, obs, act=None):
        h = self.value_dict['value_body'](obs)
        h = torch.relu(h)
        v = self.value_dict['value_head'](h)
        return v

    def get_loss(self, batch):
        action_loss, value_loss, log_p_a = self.rl.get_loss(batch, self)
        return action_loss, value_loss, log_p_a

    def get_episode(self, stat, trainer):
        info = {}
        episode = []
        state = trainer.env.reset()
        for t in range(self.args.max_steps):
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            action_out = self.policy(state_, info=info, stat=stat)
            action = select_action(self.args, action_out, status='train', info=info)
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
            episode.append(trans)
            trainer.steps += 1
            trainer.mean_reward = trainer.mean_reward + 1/trainer.steps*(np.mean(reward) - trainer.mean_reward)
            if done_:
                break
            state = next_state
        stat['mean_reward'] = trainer.mean_reward
        trainer.episodes += 1
        return episode

    def train(self, stat, trainer):
        episode = self.get_episode(stat, trainer)
        if self.args.replay:
            trainer.replay_buffer.add_experience(episode)
            replay_cond = trainer.episodes>self.args.replay_warmup\
             and len(trainer.replay_buffer.buffer)>=self.args.batch_size\
             and trainer.episodes%self.args.behaviour_update_freq==0
            if replay_cond:
                trainer.replay_process(stat)
        else:
            offline_cond = trainer.episodes%self.args.behaviour_update_freq==0
            if offline_cond:
                episode = self.Transition(*zip(*episode))
                trainer.transition_process(stat, episode)



class IndependentCommNet(CommNet):

    def __init__(self, args):
        super(IndependentCommNet, self).__init__(args)

    def identifier(self):
        if self.comm_iters == 0:
            raise RuntimeError('Please guarantee the comm iters is at least greater equal to 1.')
        elif self.comm_iters > 1:
            raise RuntimeError('Please use CommNet if the comm iters is set to the value greater than 1.')
