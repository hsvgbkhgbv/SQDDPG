from collections import namedtuple
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from util import *
from replay_buffer import *
from rl_algorithms import *


# define a transition of an episode
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'start_step', 'last_step'))

# define the hash map of rl algorithms
rl_algo_map = dict(
    reinforce=REINFORCE,
    actor_critic=ActorCritic,
    ddpg=DDPG
)



class Trainer(object):

    def __init__(self, args, model, env):
        self.args = args
        self.cuda_ = self.args.cuda and torch.cuda.is_available()
        self.behaviour_net = model(self.args).cuda() if self.cuda_ else model(self.args)
        self.rl = rl_algo_map[self.args.training_strategy](args)
        if self.args.training_strategy == 'ddpg':
            self.target_net = model(self.args).cuda() if self.cuda_ else model(self.args)
            self.target_net.load_state_dict(self.behaviour_net.state_dict())
            self.replay_buffer = ReplayBuffer(int(self.args.replay_buffer_size))
        self.env = env
        if self.args.training_strategy in ['ddpg']:
            self.action_optimizer = optim.RMSprop(self.behaviour_net.action_dict.parameters(), lr = args.lrate, alpha=0.97, eps=1e-6)
            self.value_optimizer = optim.RMSprop(self.behaviour_net.value_dict.parameters(), lr = args.lrate*args.value_coeff, alpha=0.97, eps=1e-6)
        else:
            self.optimizer = optim.RMSprop(self.behaviour_net.parameters(), lr = args.lrate, alpha=0.97, eps=1e-6)
        # self.optimizer = optim.SGD(self.behaviour_net.parameters(), lr = args.lrate)

    def get_episode(self):
        # define a stat dict
        stat = dict()
        # define the episode list
        episode = []
        # reset the environment
        state = self.env.reset()
        # define the main process of exploration
        mean_reward = []
        for t in range(self.args.max_steps):
            start_step = True if t == 0 else False
            # decide the next action and return the correlated state value (baseline)
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.args.agent_num, self.args.obs_size), self.cuda_)
            action_out = self.behaviour_net.policy(state_)
            # return the sampled actions of all of agents
            action = select_action(self.args, action_out, status='train')
            # return the rescaled (clipped) actions
            _, actual = translate_action(self.args, action)
            # receive the reward and the next state
            next_state, reward, done, _ = self.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            # define the flag of the finish of exploration
            done = done or t == self.args.max_steps - 1
            # record the mean reward for evaluation
            mean_reward.append(reward)
            # justify whether the game is done
            if done:
                last_step = True
                # record a transition
                trans = Transition(state, action.cpu().numpy(), np.array(reward), next_state, start_step, last_step)
                # trans = Transition(state, action, np.array(reward), next_state, start_step, last_step)
                episode.append(trans)
                break
            else:
                last_step = False
                # record a transition
                trans = Transition(state, action.cpu().numpy(), np.array(reward), next_state, start_step, last_step)
                # trans = Transition(state, action, np.array(reward), next_state, start_step, last_step)
                episode.append(trans)
            state = next_state
        mean_reward = np.array(mean_reward)
        mean_reward = mean_reward.mean()
        stat['num_steps'] = t + 1
        stat['mean_reward'] = mean_reward
        return (episode, stat)

    def get_batch_results(self, batch):
        stat = dict()
        if self.args.training_strategy in ['ddpg']:
            action_loss, value_loss, log_p_a = self.rl(batch, self.behaviour_net, self.target_net)
        else:
            action_loss, value_loss, log_p_a = self.rl(batch, self.behaviour_net)
        stat['action_loss'] = action_loss.item()
        stat['value_loss'] = value_loss.item()
        return stat, (action_loss, value_loss, log_p_a)

    def compute_grad(self, batch_results):
        action_loss, value_loss, log_p_a = batch_results
        loss = action_loss + self.args.value_coeff * value_loss
        if self.args.entr > 0:
            loss -= self.args.entr * multinomial_entropy(log_p_a)
        # do the backpropogation
        loss.backward()

    def action_compute_grad(self, batch_results):
        action_loss, log_p_a = batch_results
        if self.args.entr > 0:
            action_loss -= self.args.entr * multinomial_entropy(log_p_a)
        # do the backpropogation
        action_loss.backward()

    def value_compute_grad(self, batch_results):
        value_loss = batch_results
        # do the backpropogation
        value_loss.backward()

    def grad_clip(self, module):
        for name, param in module.named_parameters():
            param.grad.data.clamp_(-1, 1)

    def replay_process(self, stat):
        for i in range(self.args.replay_iters):
            batch = self.replay_buffer.get_batch_episodes(\
                                    self.args.epoch_size*self.args.max_steps)
            batch = Transition(*zip(*batch))
            s, (action_loss, value_loss, log_p_a) = self.get_batch_results(batch)
            merge_stat(s, stat)
            if self.args.training_strategy in ['ddpg']:
                self.value_optimizer.zero_grad()
                self.value_compute_grad(value_loss)
                self.grad_clip(self.behaviour_net.value_dict)
                self.value_optimizer.step()
                self.action_optimizer.zero_grad()
                self.action_compute_grad((action_loss, log_p_a))
                self.grad_clip(self.behaviour_net.action_dict)
                self.action_optimizer.step()

            else:
                self.optimizer.zero_grad()
                self.compute_grad((action_loss, value_loss, log_p_a))
                self.grad_clip(self.behaviour_net)
                self.optimizer.step()

    def run_batch(self):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while self.stats['num_episodes'] < self.args.epoch_size:
            episode, episode_stat = self.get_episode()
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode
            if self.args.training_strategy in ['ddpg']:
                self.replay_buffer.add_experience(episode)
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    def train_batch(self, t):
        batch, stat = self.run_batch()
        if self.args.training_strategy in ['ddpg']:
            self.replay_process(stat)
            if t%self.args.target_update_freq == self.args.target_update_freq - 1:
                params_target = list(self.target_net.parameters())
                params_behaviour = list(self.behaviour_net.parameters())
                for i in range(len(params_target)):
                    params_target[i] = (1 - self.args.target_lr) * params_target[i] + self.args.target_lr * params_behaviour[i]
                print ('traget net is updated!\n')
        else:
            s, (action_loss, value_loss, log_p_a) = self.get_batch_results(batch)
            merge_stat(s, stat)
            self.optimizer.zero_grad()
            self.compute_grad((action_loss, value_loss, log_p_a))
            self.grad_clip(self.behaviour_net)
            self.optimizer.step()
        return stat
