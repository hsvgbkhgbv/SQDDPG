from collections import namedtuple
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utilities.util import *
from utilities.replay_buffer import *
from learning_algorithms.actor_critic import *
from learning_algorithms.reinforce import *
from learning_algorithms.ddpg import *
from learning_algorithms.q_learning import *
from utilities.inspector import *
from arguments import *
from utilities.logger import Logger



if args.model_name in ['coma']:
    Transition = namedtuple('Transition', ('state', 'action', 'last_action', 'hidden_state', 'last_hidden_state', 'reward', 'next_state', 'done', 'last_step'))
elif args.model_name in ['ic3net']:
    Transition = namedtuple('Transition', ('state', 'action', 'last_action', 'hidden_state', 'last_hidden_state', 'reward', 'next_state', 'done', 'last_step', 'schedule'))
elif args.model_name in ['schednet']:
    Transition = namedtuple('Transition', ('state', 'action', 'last_action', 'reward', 'next_state', 'done', 'last_step', 'schedule', 'weight'))
else:
    Transition = namedtuple('Transition', ('state', 'action', 'last_action', 'reward', 'next_state', 'done', 'last_step'))



class PGTrainer(object):

    def __init__(self, args, model, env, logger, online):
        self.args = args
        self.cuda_ = self.args.cuda and torch.cuda.is_available()
        self.logger = logger
        self.online = online
        inspector(self.args)
        if self.args.target:
            target_net = model(self.args).cuda() if self.cuda_ else model(self.args)
            self.behaviour_net = model(self.args, target_net).cuda() if self.cuda_ else model(self.args, target_net)
        else:
            self.behaviour_net = model(self.args).cuda() if self.cuda_ else model(self.args)
        if self.args.replay:
            if self.online:
                self.replay_buffer = TransReplayBuffer(int(self.args.replay_buffer_size))
            else:
                self.replay_buffer = EpisodeReplayBuffer(int(self.args.replay_buffer_size))
        self.env = env
        self.action_optimizer = optim.Adam(self.behaviour_net.action_dict.parameters(), lr=args.policy_lrate)
        self.value_optimizer = optim.Adam(self.behaviour_net.value_dict.parameters(), lr=args.value_lrate)
        self.init_action = cuda_wrapper( torch.zeros(1, self.args.agent_num, self.args.action_dim), cuda=self.cuda_ )
        self.steps = 0
        self.episodes = 0
        self.mean_reward = 0

    def get_episode(self, stat):
        episode = []
        state = self.env.reset()
        info = {}
        action = self.init_action
        if self.args.model_name in ['coma']:
            info['softmax_eps'] = self.behaviour_net.eps
        if self.args.model_name in ['coma', 'ic3net']:
            self.behaviour_net.init_hidden(batch_size=1)
            last_hidden_state = self.behaviour_net.get_hidden()
        else:
            last_hidden_state = None
        for t in range(self.args.max_steps):
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.args.agent_num, self.args.obs_size), self.cuda_)
            action_ = action.clone()
            if self.args.model_name in ['ic3net']:
                gate = self.behaviour_net.gate(last_hidden_state[:, :, :self.args.hid_size])
                schedule = self.behaviour_net.schedule(gate)
            else:
                schedule = None
            action_out = self.behaviour_net.policy(state_, schedule=schedule, last_act=action_, last_hid=last_hidden_state, info=info, stat=stat)
            action = select_action(self.args, action_out, status='train', info=info)
            # return the rescaled (clipped) actions
            _, actual = translate_action(self.args, action, self.env)
            next_state, reward, done, _ = self.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            if self.args.model_name in ['coma']:
                hidden_state = self.behaviour_net.get_hidden()
                trans = Transition(state,
                                   action.cpu().numpy(),
                                   action_.cpu().numpy(),
                                   hidden_state.cpu().numpy(),
                                   last_hidden_state.cpu().numpy(),
                                   np.array(reward),
                                   next_state,
                                   done,
                                   done_
                                  )
                last_hidden_state = hidden_state
            elif self.args.model_name in ['ic3net']:
                hidden_state = self.behaviour_net.get_hidden()
                trans = Transition(state,
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
            else:
                trans = Transition(state,
                                   action.cpu().numpy(),
                                   action_.cpu().numpy(),
                                   np.array(reward),
                                   next_state,
                                   done,
                                   done_
                                  )
            episode.append(trans)
            self.steps += 1
            self.mean_reward = self.mean_reward + 1/self.steps*(np.mean(reward) - self.mean_reward)
            if done_:
                break
            state = next_state
        stat['mean_reward'] = self.mean_reward
        self.episodes += 1
        if self.args.model_name in ['coma']:
            self.behaviour_net.update_eps()
        return episode

    def train_online(self, stat):
        state = self.env.reset()
        info = {}
        action = self.init_action
        for t in range(self.args.max_steps):
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.args.agent_num, self.args.obs_size), self.cuda_)
            if self.args.model_name in ['schednet']:
                weight = self.behaviour_net.weight_generator(state_).detach()
                schedule = self.behaviour_net.weight_based_scheduler(weight)
            else:
                schedule = None
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.args.agent_num, self.args.obs_size), self.cuda_)
            action_ = action.clone()
            action_out = self.behaviour_net.policy(state_, schedule=schedule, last_act=action_, info=info, stat=stat)
            action = select_action(self.args, action_out, status='train', info=info)
            # return the rescaled (clipped) actions
            _, actual = translate_action(self.args, action, self.env)
            next_state, reward, done, _ = self.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            if self.args.model_name in ['schednet']:
                trans = Transition(state,
                                   action.cpu().numpy(),
                                   action_.cpu().numpy(),
                                   np.array(reward),
                                   next_state,
                                   done,
                                   done_,
                                   schedule.cpu().numpy(),
                                   weight.cpu().numpy()
                                  )
            else:
                trans = Transition(state,
                                   action.cpu().numpy(),
                                   action_.cpu().numpy(),
                                   np.array(reward),
                                   next_state,
                                   done,
                                   done_
                                  )
            if self.args.replay:
                self.replay_buffer.add_experience(trans)
                replay_cond = self.steps>self.args.replay_warmup\
                 and len(self.replay_buffer.buffer)>=self.args.batch_size\
                 and self.steps%self.args.behaviour_update_freq==0
                if replay_cond:
                    self.replay_process(stat)
            else:
                online_cond = self.steps%self.args.behaviour_update_freq==0
                if online_cond:
                    self.transition_process(stat, trans)
            if self.args.target:
                target_cond = self.steps%self.args.target_update_freq==0
                if target_cond:
                    self.behaviour_net.update_target()
            self.steps += 1
            self.mean_reward = self.mean_reward + 1/self.steps*(np.mean(reward) - self.mean_reward)
            stat['mean_reward'] = self.mean_reward
            if done_:
                break
            state = next_state
        self.episodes += 1
        if self.args.model_name in ['coma']:
            self.behaviour_net.update_eps()

    def train_offline(self, stat):
        episode = self.get_episode(stat)
        if self.args.replay:
            self.replay_buffer.add_experience(episode)
            replay_cond = self.episodes>self.args.replay_warmup\
             and len(self.replay_buffer.buffer)>=self.args.batch_size\
             and self.episodes%self.args.behaviour_update_freq==0
            if replay_cond:
                self.replay_process(stat)
        else:
            offline_cond = self.episodes%self.args.behaviour_update_freq==0
            if offline_cond:
                episode = Transition(*zip(*episode))
                self.transition_process(stat, episode)

    def get_loss(self, batch):
        action_loss, value_loss, log_p_a = self.behaviour_net.get_loss(batch)
        return action_loss, value_loss, log_p_a

    def action_compute_grad(self, stat, loss):
        action_loss, log_p_a = loss
        if not self.args.continuous:
            if self.args.entr > 0:
                entropy = multinomial_entropy(log_p_a)
                action_loss -= self.args.entr * entropy
                stat['entropy'] = entropy.item()
        action_loss.backward()

    def value_compute_grad(self, batch_loss):
        value_loss = batch_loss
        value_loss.backward()

    def grad_clip(self, module):
        for name, param in module.named_parameters():
            param.grad.data.clamp_(-1, 1)

    def replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = Transition(*zip(*batch))
        self.transition_process(stat, batch)

    def transition_process(self, stat, trans):
        action_loss, value_loss, log_p_a = self.get_loss(trans)
        self.value_optimizer.zero_grad()
        self.value_compute_grad(value_loss)
        if self.args.grad_clip:
            self.grad_clip(self.behaviour_net.value_dict)
        stat['value_grad_norm'] = get_grad_norm(self.behaviour_net.value_dict)
        self.value_optimizer.step()
        stat['value_loss'] = value_loss.item()
        self.action_optimizer.zero_grad()
        self.action_compute_grad(stat, (action_loss, log_p_a))
        if self.args.grad_clip:
            self.grad_clip(self.behaviour_net.action_dict)
        stat['policy_grad_norm'] = get_grad_norm(self.behaviour_net.action_dict)
        self.action_optimizer.step()
        stat['action_loss'] = action_loss.item()

    def run(self, stat):
        if self.online:
            self.train_online(stat)
        else:
            self.train_offline(stat)

    def record(self, stat):
        for tag, value in stat.items():
            if isinstance(value, np.ndarray):
                self.logger.image_summary(tag, value, self.episodes)
            else:
                self.logger.scalar_summary(tag, value, self.episodes)

    def print_info(self, stat):
        action_loss = stat.get('action_loss', 0)
        value_loss = stat.get('value_loss', 0)
        print ('This is the episode: {}, the mean reward is {:2.4f}, the current action loss is {:2.4f} and the current value loss is: {:2.4f}\n'\
        .format(self.episodes, stat['mean_reward'], action_loss, value_loss))



class QTrainer(object):

    def __init__(self, args, model, env, logger):
        self.args = args
        self.cuda_ = self.args.cuda and torch.cuda.is_available()
        self.logger = logger
        inspector(self.args)
        if self.args.target:
            target_net = model(self.args).cuda() if self.cuda_ else model(self.args)
            self.behaviour_net = model(self.args, target_net).cuda() if self.cuda_ else model(self.args, target_net)
        else:
            self.behaviour_net = model(self.args).cuda() if self.cuda_ else model(self.args)
        if self.args.replay:
            self.replay_buffer = TransReplayBuffer(int(self.args.replay_buffer_size))
        self.env = env
        self.value_optimizer = optim.Adam(self.behaviour_net.value_dict.parameters(), lr=args.value_lrate)
        self.init_action = cuda_wrapper( torch.zeros(1, self.args.agent_num, self.args.action_dim), cuda=self.cuda_ )
        self.steps = 0
        self.episodes = 0
        self.mean_reward = 0

    def train_online(self, stat):
        state = self.env.reset()
        info = {}
        action = self.init_action
        for t in range(self.args.max_steps):
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.args.agent_num, self.args.obs_size), self.cuda_)
            action_ = action.clone()
            action_value = self.behaviour_net.value(state_, action_, info=info, stat=stat)
            action = select_action(self.args, action_value, status='train', info=info)
            # return the rescaled (clipped) actions
            _, actual = translate_action(self.args, action, self.env)
            next_state, reward, done, _ = self.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            trans = Transition(state,
                               action.cpu().numpy(),
                               action_.cpu().numpy(),
                               np.array(reward),
                               next_state,
                               done,
                               done_
                              )
            if self.args.replay:
                self.replay_buffer.add_experience(trans)
                replay_cond = self.steps>self.args.replay_warmup\
                 and len(self.replay_buffer.buffer)>=self.args.batch_size\
                 and self.steps%self.args.behaviour_update_freq==0
                if replay_cond:
                    self.replay_process(stat)
            else:
                online_cond = self.steps%self.args.behaviour_update_freq==0
                if online_cond:
                    self.transition_process(stat, trans)
            if self.args.target:
                target_cond = self.steps%self.args.target_update_freq==0
                if target_cond:
                    self.behaviour_net.update_target()
            self.steps += 1
            self.mean_reward = self.mean_reward + 1/self.steps*(np.mean(reward) - self.mean_reward)
            stat['mean_reward'] = self.mean_reward
            if done_:
                break
            state = next_state
        self.episodes += 1

    def get_loss(self, batch):
        value_loss = self.behaviour_net.get_loss(batch)
        return value_loss

    def value_compute_grad(self, batch_loss):
        value_loss = batch_loss
        value_loss.backward()

    def grad_clip(self, module):
        for name, param in module.named_parameters():
            param.grad.data.clamp_(-1, 1)

    def replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = Transition(*zip(*batch))
        self.transition_process(stat, batch)

    def transition_process(self, stat, trans):
        value_loss = self.get_loss(trans)
        self.value_optimizer.zero_grad()
        self.value_compute_grad(value_loss)
        if self.args.grad_clip:
            self.grad_clip(self.behaviour_net.value_dict)
        stat['value_grad_norm'] = get_grad_norm(self.behaviour_net.value_dict)
        self.value_optimizer.step()
        stat['value_loss'] = value_loss.item()

    def run(self):
        stat = dict()
        self.train_online(stat)

    def record(self, stat):
        for tag, value in stat.items():
            if isinstance(value, np.ndarray):
                self.logger.image_summary(tag, value, self.episodes)
            else:
                self.logger.scalar_summary(tag, value, self.episodes)

    def print_info(self, stat):
        value_loss = stat.get('value_loss', 0)
        print ('This is the episode: {}, the mean reward is {:2.4f} and the current value loss is: {:2.4f}\n'\
        .format(self.episodes, stat['mean_reward'], value_loss))
