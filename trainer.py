from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from util import *
from replay_buffer import *


# define a transition of an episode
Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state', 'reward', 'misc'))


class Trainer(object):

    def __init__(self, args, policy_net, env, replay):
        self.args = args
        self.policy_net = policy_net
        self.policy_net
        self.env = env
        self.optimizer = optim.RMSprop(policy_net.parameters(), lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]
        self.replay = replay
        if self.replay:
            self.replay_buffer = ReplayBuffer(int(1e7), 0.2)

    def get_episode(self):
        # define the episode list
        episode = []
        # reset the environment
        state = self.env.reset()
        # set up two auxilliary dictionaries
        stat = dict()
        info = dict()
        # define the main process of exploration
        mean_reward = []
        for t in range(self.args.max_steps):
            misc = dict()
            # decide the next action and return the correlated state value (baseline)
            action_out, value = self.policy_net.action(state, info)
            # return the sampled actions of all of agents
            action = select_action(self.args, action_out, 'train')
            # return the rescaled (clipped) actions
            _, actual = translate_action(self.args, self.env, action)
            # receive the reward and the next state
            next_state, reward, done, info = self.env.step(actual)
            done = np.sum(done)
            # record the alive agents
            if 'alive_mask' in info:
                # serve for the starcraft environment
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)
            # define the flag of the finish of exploration
            done = done or t == self.args.max_steps - 1

            reward = np.array(reward)
            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)
            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                # serve for traffic environment
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)
            # record a transition
            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            # record the current transition to the whole episode
            episode.append(trans)

            state = next_state

            mean_reward.append(reward)

            if done:
                mean_reward = np.array(mean_reward)
                mean_reward = mean_reward.mean()
                break
        stat['num_steps'] = t + 1
        stat['mean_reward'] = mean_reward
        stat['steps_taken'] = stat['num_steps']
        return (episode, stat)

    def compute_grad(self, batch):

        stat = dict()

        action_dim = self.args.action_dim
        n = self.args.agent_num
        batch_size = len(batch.state)

        with torch.no_grad():
            rewards = torch.Tensor(batch.reward)
            episode_masks = torch.Tensor(batch.episode_mask)
            episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
            batch_action = torch.stack(batch.action, dim=0).float()
            actions = torch.Tensor(batch_action)
            actions = actions.transpose(1, 2).view(-1, n, action_dim)

        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        if self.args.decomposition:
            action_out = [torch.cat(a, dim=0).contiguous().view(-1,action_dim) for a in action_out]
        else:
            action_out = [torch.cat(a, dim=0) for a in action_out]

        with torch.no_grad():
            alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)
            coop_returns = torch.Tensor(batch_size, n)
            ncoop_returns = torch.Tensor(batch_size, n)
            returns = torch.Tensor(batch_size, n)
        # deltas = torch.Tensor(batch_size, n)
        with torch.no_grad():
            advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        # calculate the return reversely and the reward is shared
        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i]
            prev_ncoop_return = ncoop_returns[i]

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                         + ((1 - self.args.mean_ratio) * ncoop_returns[i])

        # calculate the advantage
        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        # normalize the advantage
        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        # take the policy of the actions
        if self.args.continuous:
            actions = actions.contiguous().view(-1, self.args.action_dim)
            if self.args.decomposition:
                log_prob = []
                action_means, action_log_stds, action_stds = action_out
                for i in range(action_dim):
                    log_prob.append(normal_log_density(actions[:, i:i+1], action_means[:, i:i+1], action_log_stds[:, i:i+1], action_stds[:, i:i+1]))
                log_prob = log_prob[0] * log_prob[1]
            else:
                action_means, action_log_stds, action_stds = action_out
                log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = action_out
            actions = actions.contiguous().view(-1, 1)
            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob / self.args.batch_size
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze() / self.args.batch_size
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1) / self.args.batch_size
        value_loss *= alive_masks
        value_loss = value_loss.sum()
        stat['value_loss'] = value_loss.item()

        loss = action_loss + self.args.value_coeff * value_loss

        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy

        loss.backward()

        return stat

    def run_batch(self):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode()
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode
        if self.replay:
            self.replay_buffer.add_experience(episode)
        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    def train_batch(self):
        batch, stat = self.run_batch()
        self.optimizer.zero_grad()
        s = self.compute_grad(batch)
        merge_stat(s, stat)
        # for p in self.params:
        #     if p._grad is not None:
        #         p._grad.data /= stat['num_steps']
        self.optimizer.step()
        # if self.replay:
        #     for i in range(20):
        #         batch = self.replay_buffer.get_batch_episodes(\
        #                                 self.args.batch_size)
        #         batch = Transition(*zip(*batch))
        #         self.optimizer.zero_grad()
        #         s = self.compute_grad(batch)
        #         self.optimizer.step()
        #     print ('Finish replay in 20 iterations!')
        return stat
