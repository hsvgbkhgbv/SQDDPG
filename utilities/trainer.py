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



if args.model_name in ['coma']:
    Transition = namedtuple('Transition', ('state', 'action', 'last_action', 'hidden_state', 'last_hidden_state', 'reward', 'next_state', 'done', 'last_step'))
else:
    Transition = namedtuple('Transition', ('state', 'action', 'last_action', 'reward', 'next_state', 'done', 'last_step'))



class PGOfflineTrainer(object):

    def __init__(self, args, model, env):
        self.args = args
        self.cuda_ = self.args.cuda and torch.cuda.is_available()
        inspector(self.args)
        if self.args.target:
            target_net = model(self.args).cuda() if self.cuda_ else model(self.args)
            self.behaviour_net = model(self.args, target_net).cuda() if self.cuda_ else model(self.args, target_net)
        else:
            self.behaviour_net = model(self.args).cuda() if self.cuda_ else model(self.args)
        if self.args.replay:
            self.replay_buffer = ReplayBuffer(int(self.args.replay_buffer_size))
        self.env = env
        self.action_optimizer = optim.Adam(self.behaviour_net.action_dict.parameters(), lr=args.policy_lrate)
        self.value_optimizer = optim.Adam(self.behaviour_net.value_dict.parameters(), lr=args.value_lrate)
        self.init_action = cuda_wrapper( torch.zeros(1, self.args.agent_num, self.args.action_dim), cuda=self.cuda_ )
        self.steps = 0
        
    def get_episode(self, stat):
        episode = []
        state = self.env.reset()
        mean_reward = []
        info = {}
        action = self.init_action
        if self.args.epsilon_softmax:
            info['softmax_eps'] = self.behaviour_net.eps
        if self.args.model_name in ['coma']:
            info['get_episode'] = True
            self.behaviour_net.init_hidden(batch_size=1)
            last_hidden_state = self.behaviour_net.get_hidden()
        else:
            last_hidden_state = None
        for t in range(self.args.max_steps):
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.args.agent_num, self.args.obs_size), self.cuda_)
            action_ = action.clone()
            action_out = self.behaviour_net.policy(state_, action_, last_hid=last_hidden_state, info=info, stat=stat)
            action = select_action(self.args, action_out, status='train', info=info)
            # return the rescaled (clipped) actions
            _, actual = translate_action(self.args, action, self.env)
            next_state, reward, done, _ = self.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            mean_reward.append(reward)
            if self.args.model_name in ['coma']:
                hidden_state = self.behaviour_net.get_hidden()
                trans = Transition(state, action.cpu().numpy(), action_.cpu().numpy(), hidden_state.cpu().numpy(), last_hidden_state.cpu().numpy(), np.array(reward), next_state, done, done_)
                last_hidden_state = hidden_state
            else:
                trans = Transition(state, action.cpu().numpy(), action_.cpu().numpy(), np.array(reward), next_state, done, done_)
            episode.append(trans)
            self.steps += 1
            if done_:
                break
            state = next_state
        num_steps = t+1
        mean_reward = np.mean(mean_reward)
        if self.args.epsilon_softmax:
            self.behaviour_net.update_eps()
        return episode, mean_reward, num_steps

    def get_batch_loss(self, batch):
        action_loss, value_loss, log_p_a = self.behaviour_net.get_loss(batch)
        return action_loss, value_loss, log_p_a

    def action_compute_grad(self, stat, batch_loss):
        action_loss, log_p_a = batch_loss
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
        action_loss_ = 0
        value_loss_ = 0
        policy_grad_norm = 0
        value_grad_norm = 0
        for i in range(self.args.replay_iters):
            batch = self.replay_buffer.get_batch(self.args.batch_size)
            batch = Transition(*zip(*batch))
            action_loss, value_loss, log_p_a = self.get_batch_loss(batch)
            action_loss_ += action_loss.item()
            value_loss_ += value_loss.item()
            self.value_optimizer.zero_grad()
            self.value_compute_grad(value_loss)
            if self.args.grad_clip:
                self.grad_clip(self.behaviour_net.value_dict)
            value_grad_norm += get_grad_norm(self.behaviour_net.value_dict)
            if self.steps > self.args.replay_warmup and len(self.replay_buffer.buffer) >= self.args.batch_size:
                self.value_optimizer.step()
            merge_dict(stat, 'value_loss', value_loss_ / self.args.replay_iters)
            merge_dict(stat, 'value_grad_norm', value_grad_norm / self.args.replay_iters)
            self.action_optimizer.zero_grad()
            self.action_compute_grad(stat, (action_loss, log_p_a))
            if self.args.grad_clip:
                self.grad_clip(self.behaviour_net.action_dict)
            policy_grad_norm += get_grad_norm(self.behaviour_net.action_dict)
            if self.steps > self.args.replay_warmup and len(self.replay_buffer.buffer) >= self.args.batch_size:
                self.action_optimizer.step()
            merge_dict(stat, 'action_loss', action_loss_ / self.args.replay_iters)
            merge_dict(stat, 'policy_grad_norm', policy_grad_norm / self.args.replay_iters)

    def offline_process(self, stat, batch):
        action_loss, value_loss, log_p_a = self.get_batch_loss(batch)
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

    def run_batch(self):
        batch = []
        stats = dict()
        num_episodes = 0
        average_mean_reward = 0
        average_num_steps = 0
        while num_episodes < self.args.epoch_size:
            episode, mean_reward, num_steps = self.get_episode(stats)
            average_mean_reward += mean_reward
            average_num_steps += num_steps
            num_episodes += 1
            batch += episode
            if self.args.replay:
                self.replay_buffer.add_experience(episode)
        stats['mean_reward'] = average_mean_reward / self.args.epoch_size
        stats['average_episode_steps'] = average_num_steps / self.args.epoch_size
        batch = Transition(*zip(*batch))
        stat = self.train_batch(batch, stats)
        return stat

    def train_batch(self, batch, stat):
        if self.args.model_name in ['maddpg', 'mfac']:
            self.replay_process(stat)
        else:
            self.offline_process(stat, batch)
            if self.args.replay:
                self.replay_process(stat)
        if self.args.target:
            if self.steps%self.args.target_update_freq == self.args.target_update_freq-1:
                self.behaviour_net.update_target()
        return stat



class PGOnlineTrainer(object):

    def __init__(self, args, model, env):
        self.args = args
        self.cuda_ = self.args.cuda and torch.cuda.is_available()
        inspector(self.args)
        if self.args.target:
            target_net = model(self.args).cuda() if self.cuda_ else model(self.args)
            self.behaviour_net = model(self.args, target_net).cuda() if self.cuda_ else model(self.args, target_net)
        else:
            self.behaviour_net = model(self.args).cuda() if self.cuda_ else model(self.args)
        if self.args.replay:
            self.replay_buffer = ReplayBuffer(int(self.args.replay_buffer_size))
        self.env = env
        self.action_optimizer = optim.Adam(self.behaviour_net.action_dict.parameters(), lr=args.policy_lrate)
        self.value_optimizer = optim.Adam(self.behaviour_net.value_dict.parameters(), lr=args.value_lrate)
        self.init_action = cuda_wrapper( torch.zeros(1, self.args.agent_num, self.args.action_dim), cuda=self.cuda_ )
        self.steps = 0
        
    def train_online(self, stat):
        episode = []
        state = self.env.reset()
        mean_reward = []
        info = {}
        action = self.init_action
        if self.args.epsilon_softmax:
            info['softmax_eps'] = self.behaviour_net.eps
        if self.args.model_name in ['coma']:
            info['get_episode'] = True
            self.behaviour_net.init_hidden(batch_size=1)
            last_hidden_state = self.behaviour_net.get_hidden()
        else:
            last_hidden_state = None
        for t in range(self.args.max_steps):
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.args.agent_num, self.args.obs_size), self.cuda_)
            action_ = action.clone()
            action_out = self.behaviour_net.policy(state_, action_, last_hid=last_hidden_state, info=info, stat=stat)
            action = select_action(self.args, action_out, status='train', info=info)
            # return the rescaled (clipped) actions
            _, actual = translate_action(self.args, action, self.env)
            next_state, reward, done, _ = self.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            mean_reward.append(reward)
            if self.args.model_name in ['coma']:
                hidden_state = self.behaviour_net.get_hidden()
                trans = Transition(state, action.cpu().numpy(), action_.cpu().numpy(), hidden_state.cpu().numpy(), last_hidden_state.cpu().numpy(), np.array(reward), next_state, done, done_)
                last_hidden_state = hidden_state
            else:
                trans = Transition(state, action.cpu().numpy(), action_.cpu().numpy(), np.array(reward), next_state, done, done_)
            if self.args.replay:
                self.replay_buffer.add_experience([trans])
                if self.steps > self.args.replay_warmup and len(self.replay_buffer.buffer) >= self.args.batch_size and self.steps%self.args.behaviour_update_freq == self.args.behaviour_update_freq-1:
                    update_flag = True
                else:
                    update_flag = False
                self.replay_process(stat, update_flag)
            else:
                if self.steps%self.args.behaviour_update_freq == self.args.behaviour_update_freq-1:
                    update_flag = True
                else:
                    update_flag = False
                self.online_process(stat, trans, update_flag)
            if self.args.target:
                if self.steps%self.args.target_update_freq == self.args.target_update_freq-1:
                    self.behaviour_net.update_target()
            self.steps += 1
            if done_:
                break
            state = next_state
        num_steps = t+1
        mean_reward = np.mean(mean_reward)
        if self.args.epsilon_softmax:
            self.behaviour_net.update_eps()
        return mean_reward, num_steps

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

    def replay_process(self, stat, update_flag):
        action_loss_ = 0
        value_loss_ = 0
        policy_grad_norm = 0
        value_grad_norm = 0
        for i in range(self.args.replay_iters):
            if update_flag:
                batch = self.replay_buffer.get_batch(self.args.batch_size)
                batch = Transition(*zip(*batch))
                action_loss, value_loss, log_p_a = self.get_loss(batch)
                action_loss_ += action_loss.item()
                value_loss_ += value_loss.item()
                self.value_optimizer.zero_grad()
                self.value_compute_grad(value_loss)
                if self.args.grad_clip:
                    self.grad_clip(self.behaviour_net.value_dict)
                value_grad_norm += get_grad_norm(self.behaviour_net.value_dict)
                self.value_optimizer.step()
                self.action_optimizer.zero_grad()
                self.action_compute_grad(stat, (action_loss, log_p_a))
                if self.args.grad_clip:
                    self.grad_clip(self.behaviour_net.action_dict)
                policy_grad_norm += get_grad_norm(self.behaviour_net.action_dict)
                self.action_optimizer.step()
            merge_dict(stat, 'value_loss', value_loss_ / self.args.replay_iters)
            merge_dict(stat, 'value_grad_norm', value_grad_norm / self.args.replay_iters)
            merge_dict(stat, 'action_loss', action_loss_ / self.args.replay_iters)
            merge_dict(stat, 'policy_grad_norm', policy_grad_norm / self.args.replay_iters)

    def online_process(self, stat, trans, update_flag):
        action_loss, value_loss, log_p_a = self.get_loss(trans)
        self.value_optimizer.zero_grad()
        self.value_compute_grad(value_loss)
        if self.args.grad_clip:
            self.grad_clip(self.behaviour_net.value_dict)
        stat['value_grad_norm'] = get_grad_norm(self.behaviour_net.value_dict)
        if update_flag:
            self.value_optimizer.step()
        stat['value_loss'] = value_loss.item()
        self.action_optimizer.zero_grad()
        self.action_compute_grad(stat, (action_loss, log_p_a))
        if self.args.grad_clip:
            self.grad_clip(self.behaviour_net.action_dict)
        stat['policy_grad_norm'] = get_grad_norm(self.behaviour_net.action_dict)
        if update_flag:
            self.action_optimizer.step()
        stat['action_loss'] = action_loss.item()

    def run_batch(self):
        batch = []
        stats = dict()
        num_episodes = 0
        average_mean_reward = 0
        average_num_steps = 0
        while num_episodes < self.args.epoch_size:
            mean_reward, num_steps = self.train_online(stats)
            average_mean_reward += mean_reward
            average_num_steps += num_steps
            num_episodes += 1
        stats['mean_reward'] = average_mean_reward / self.args.epoch_size
        stats['average_episode_steps'] = average_num_steps / self.args.epoch_size
        return stats



class QOfflineTrainer(object):

    def __init__(self, args, model, env):
        self.args = args
        self.cuda_ = self.args.cuda and torch.cuda.is_available()
        inspector(self.args)
        if self.args.target:
            target_net = model(self.args).cuda() if self.cuda_ else model(self.args)
            self.behaviour_net = model(self.args, target_net).cuda() if self.cuda_ else model(self.args, target_net)
        else:
            self.behaviour_net = model(self.args).cuda() if self.cuda_ else model(self.args)
        if self.args.replay:
            self.replay_buffer = ReplayBuffer(int(self.args.replay_buffer_size))
        self.env = env
        self.value_optimizer = optim.Adam(self.behaviour_net.value_dict.parameters(), lr=args.value_lrate)
        self.init_action = cuda_wrapper( torch.zeros(1, self.args.agent_num, self.args.action_dim), cuda=self.cuda_ )
        self.steps = 0
        
    def get_episode(self, stat):
        episode = []
        state = self.env.reset()
        mean_reward = []
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
            mean_reward.append(reward)
            trans = Transition(state, action.cpu().numpy(), action_.cpu().numpy(), np.array(reward), next_state, done, done_)
            episode.append(trans)
            self.steps += 1
            if done_:
                break
            state = next_state
        num_steps = t+1
        mean_reward = np.mean(mean_reward)
        return episode, mean_reward, num_steps

    def get_batch_loss(self, batch):
        value_loss = self.behaviour_net.get_loss(batch)
        return value_loss

    def value_compute_grad(self, batch_loss):
        value_loss = batch_loss
        value_loss.backward()

    def grad_clip(self, module):
        for name, param in module.named_parameters():
            param.grad.data.clamp_(-1, 1)

    def replay_process(self, stat):
        value_loss_ = 0
        value_grad_norm = 0
        for i in range(self.args.replay_iters):
            batch = self.replay_buffer.get_batch(self.args.batch_size)
            batch = Transition(*zip(*batch))
            value_loss = self.get_batch_loss(batch)
            value_loss_ += value_loss.item()
            self.value_optimizer.zero_grad()
            self.value_compute_grad(value_loss)
            if self.args.grad_clip:
                self.grad_clip(self.behaviour_net.value_dict)
            value_grad_norm += get_grad_norm(self.behaviour_net.value_dict)
            if self.steps > self.args.replay_warmup and len(self.replay_buffer.buffer) >= self.args.batch_size:
                self.value_optimizer.step()
            merge_dict(stat, 'value_loss', value_loss_ / self.args.replay_iters)
            merge_dict(stat, 'value_grad_norm', value_grad_norm / self.args.replay_iters)

    def offline_process(self, stat, batch):
        value_loss = self.get_batch_loss(batch)
        self.value_optimizer.zero_grad()
        self.value_compute_grad(value_loss)
        if self.args.grad_clip:
            self.grad_clip(self.behaviour_net.value_dict)
        stat['value_grad_norm'] = get_grad_norm(self.behaviour_net.value_dict)
        self.value_optimizer.step()
        stat['value_loss'] = value_loss.item()

    def run_batch(self):
        batch = []
        stats = dict()
        num_episodes = 0
        average_mean_reward = 0
        average_num_steps = 0
        while num_episodes < self.args.epoch_size:
            episode, mean_reward, num_steps = self.get_episode(stats)
            average_mean_reward += mean_reward
            average_num_steps += num_steps
            num_episodes += 1
            batch += episode
            if self.args.replay:
                self.replay_buffer.add_experience(episode)
        stats['mean_reward'] = average_mean_reward / self.args.epoch_size
        stats['average_episode_steps'] = average_num_steps / self.args.epoch_size
        batch = Transition(*zip(*batch))
        stat = self.train_batch(batch, stats)
        return stat

    def train_batch(self, batch, stat):
        self.offline_process(stat, batch)
        if self.args.replay:
            self.replay_process(stat)
        if self.args.target:
            if self.steps%self.args.target_update_freq == self.args.target_update_freq-1:
                self.behaviour_net.update_target()
        return stat



class QOnlineTrainer(object):

    def __init__(self, args, model, env):
        self.args = args
        self.cuda_ = self.args.cuda and torch.cuda.is_available()
        inspector(self.args)
        if self.args.target:
            target_net = model(self.args).cuda() if self.cuda_ else model(self.args)
            self.behaviour_net = model(self.args, target_net).cuda() if self.cuda_ else model(self.args, target_net)
        else:
            self.behaviour_net = model(self.args).cuda() if self.cuda_ else model(self.args)
        if self.args.replay:
            self.replay_buffer = ReplayBuffer(int(self.args.replay_buffer_size))
        self.env = env
        self.value_optimizer = optim.Adam(self.behaviour_net.value_dict.parameters(), lr=args.value_lrate)
        self.init_action = cuda_wrapper( torch.zeros(1, self.args.agent_num, self.args.action_dim), cuda=self.cuda_ )
        self.steps = 0
        
    def train_online(self, stat):
        episode = []
        state = self.env.reset()
        mean_reward = []
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
            mean_reward.append(reward)
            trans = Transition(state, action.cpu().numpy(), action_.cpu().numpy(), np.array(reward), next_state, done, done_)
            if self.args.replay:
                self.replay_buffer.add_experience([trans])
                if self.steps > self.args.replay_warmup and len(self.replay_buffer.buffer) >= self.args.batch_size and self.steps%self.args.behaviour_update_freq == self.args.behaviour_update_freq-1:
                    update_flag = True
                else:
                    update_flag = False
                    self.replay_process(stat, update_flag)
            else:
                if self.steps%self.args.behaviour_update_freq == self.args.behaviour_update_freq-1:
                    update_flag = True
                else:
                    update_flag = False
                self.online_process(stat, trans, update_flag)
            if self.args.target:
                if self.steps%self.args.target_update_freq == self.args.target_update_freq-1:
                    self.behaviour_net.update_target()
            self.steps += 1
            if done_:
                break
            state = next_state
        num_steps = t+1
        mean_reward = np.mean(mean_reward)
        return mean_reward, num_steps

    def get_loss(self, batch):
        value_loss = self.behaviour_net.get_loss(batch)
        return value_loss

    def value_compute_grad(self, batch_loss):
        value_loss = batch_loss
        value_loss.backward()

    def grad_clip(self, module):
        for name, param in module.named_parameters():
            param.grad.data.clamp_(-1, 1)

    def replay_process(self, stat, update_flag):
        value_loss_ = 0
        value_grad_norm = 0
        for i in range(self.args.replay_iters):
            if update_flag:
                batch = self.replay_buffer.get_batch(self.args.batch_size)
                batch = Transition(*zip(*batch))
                value_loss = self.get_loss(batch)
                value_loss_ += value_loss.item()
                self.value_optimizer.zero_grad()
                self.value_compute_grad(value_loss)
                if self.args.grad_clip:
                    self.grad_clip(self.behaviour_net.value_dict)
                value_grad_norm += get_grad_norm(self.behaviour_net.value_dict)
                if update_flag:
                    self.value_optimizer.step()
            merge_dict(stat, 'value_loss', value_loss_ / self.args.replay_iters)
            merge_dict(stat, 'value_grad_norm', value_grad_norm / self.args.replay_iters)

    def online_process(self, stat, trans, update_flag):
        value_loss = self.get_batch_loss(trans)
        self.value_optimizer.zero_grad()
        self.value_compute_grad(value_loss)
        if self.args.grad_clip:
            self.grad_clip(self.behaviour_net.value_dict)
        stat['value_grad_norm'] = get_grad_norm(self.behaviour_net.value_dict)
        if update_flag:
            self.value_optimizer.step()
        stat['value_loss'] = value_loss.item()

    def run_batch(self):
        batch = []
        stats = dict()
        num_episodes = 0
        average_mean_reward = 0
        average_num_steps = 0
        while num_episodes < self.args.epoch_size:
            mean_reward, num_steps = self.train_online(stats)
            average_mean_reward += mean_reward
            average_num_steps += num_steps
            num_episodes += 1
        stats['mean_reward'] = average_mean_reward / self.args.epoch_size
        stats['average_episode_steps'] = average_num_steps / self.args.epoch_size
        return stats
