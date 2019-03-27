from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from util import *
from replay_buffer import *
import torch.multiprocessing as mp


# define a transition of an episode
Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state', 'next_value', 'reward', 'misc'))


class Trainer(mp.Process):

    def __init__(self, args, policy_net, env, replay):
        super(Trainer, self).__init__()
        self.args = args
        self.policy_net = policy_net.cuda() if torch.cuda.is_available() else policy_net
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

            misc['start_step'] = True if t == 0 else False

            # decide the next action and return the correlated state value (baseline)
            action_out, value = self.policy_net(state, info)
            # return the sampled actions of all of agents
            action = select_action(self.args, action_out, 'train')
            # return the rescaled (clipped) actions
            _, actual = translate_action(self.args, self.env, action)
            if self.args.training_strategy == 'actor_critic':
                act = action.squeeze()
                value = torch.cat([value[:, i, act[i]].unsqueeze(-1) for i in range(act.size(0))], dim=-1)

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

            # take the next value or action value
            next_action_out, next_value = self.policy_net.action(next_state, info)
            if self.args.training_strategy == 'actor_critic':
                next_action = select_action(self.args, next_action_out, 'train')
                next_act = next_action.squeeze()
                next_value = torch.cat([next_value[:, i, next_act[i]].unsqueeze(-1) for i in range(act.size(0))], dim=-1)

            mean_reward.append(reward)

            if done:
                misc['last_step'] = True
                # record a transition
                trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, next_value, reward, misc)
                # record the current transition to the whole episode
                episode.append(trans)
                break
            else:
                misc['last_step'] = False
                # record a transition
                trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, next_value, reward, misc)
                # record the current transition to the whole episode
                episode.append(trans)

            state = next_state

        mean_reward = np.array(mean_reward)
        mean_reward = mean_reward.mean()
        stat['num_steps'] = t + 1
        stat['mean_reward'] = mean_reward
        stat['steps_taken'] = stat['num_steps']

        return (episode, stat)

    def compute_grad(self, batch):

        stat = dict()

        action_dim = self.args.action_dim
        n = self.args.agent_num
        batch_size = len(batch.state)

        # define some necessary containers
        with torch.no_grad():
            rewards = torch.Tensor(batch.reward)
            episode_masks = torch.Tensor(batch.episode_mask)
            episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
            actions = torch.stack(batch.action, dim=0).float()
            actions = actions.transpose(1, 2).view(-1, n, 1)
            if torch.cuda.is_available():
                rewards = rewards.cuda()
                episode_masks = episode_masks.cuda()
                episode_mini_masks = episode_mini_masks.cuda()
        values = torch.cat(batch.value, dim=0)
        next_values = torch.cat(batch.next_value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]
        with torch.no_grad():
            alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)
            coop_returns = torch.Tensor(batch_size, n)
            returns = torch.Tensor(batch_size, n)
            if self.args.training_strategy == 'actor_critic':
                deltas = torch.Tensor(batch_size, n)
                if torch.cuda.is_available():
                    deltas = deltas.cuda()
            advantages = torch.Tensor(batch_size, n)
            if torch.cuda.is_available():
                alive_masks = alive_masks.cuda()
                coop_returns = coop_returns.cuda()
                returns = returns.cuda()
                advantages = advantages.cuda()
            values = values.view(batch_size, n)
            next_values = next_values.view(batch_size, n)

        # calculate the returns or estimated returns
        if self.args.training_strategy == 'reinforce':
            # calculate the return reversely and the reward is shared
            for i in reversed(range(rewards.size(0))):
                if batch.misc[i]['last_step']:
                    prev_coop_return = 0
                coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
                prev_coop_return = coop_returns[i]
                returns[i] = coop_returns[i].mean()
        elif self.args.training_strategy == 'actor_critic':
            # calculate the estimated action value
            for i in range(rewards.size(0)):
                if batch.misc[i]['start_step']:
                    I = 1
                coop_returns[i] = values[i] * episode_masks[i]
                deltas[i] = I * (rewards[i] + self.args.gamma * next_values[i].detach() * episode_masks[i] - coop_returns[i]).mean()
                # returns[i] = I * coop_returns[i].mean()
                returns[i] = deltas[i]
                I *= self.args.gamma
        # calculate the advantage
        for i in reversed(range(rewards.size(0))):
            if self.args.training_strategy == 'reinforce':
                advantages[i] = returns[i] - values.data[i]
            elif self.args.training_strategy == 'actor_critic':
                advantages[i] = returns[i]

        # normalize the advantage
        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        # take the policy of actions
        if self.args.continuous:
            actions = actions.contiguous().view(-1, self.args.action_dim)
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = action_out
            actions = actions.contiguous().view(-1, 1)
            log_prob = multinomials_log_density(actions, log_p_a)

        # calculate the advantages
        action_loss = -advantages.view(-1) * log_prob.squeeze() / self.args.batch_size
        action_loss *= alive_masks
        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # calculate the value loss
        if self.args.training_strategy == 'reinforce':
            targets = returns
            value_loss = (values - targets).pow(2).view(-1) / self.args.batch_size
        elif self.args.training_strategy == 'actor_critic':
            value_loss = deltas.pow(2).view(-1) / self.args.batch_size
        value_loss *= alive_masks
        value_loss = value_loss.sum()
        stat['value_loss'] = value_loss.item()

        # combine the policy objective function and the value loss together
        loss = action_loss + self.args.value_coeff * value_loss

        # entropy regularization term, but it is obly available to discrete policy
        if not self.args.continuous:
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy

        # do the backpropogation
        if self.replay:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

        return stat

    def run_batch(self):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while self.stats['num_episodes'] < self.args.batch_size:
            episode, episode_stat = self.get_episode()
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            # maybe here
            batch += episode
            if self.replay:
                self.replay_buffer.add_experience(episode)
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    def train_batch(self):
        batch, stat = self.run_batch()
        self.optimizer.zero_grad()
        s = self.compute_grad(batch)
        merge_stat(s, stat)
        self.optimizer.step()
        if self.replay:
            for i in range(20):
                batch = self.replay_buffer.get_batch_episodes(\
                                        self.args.batch_size)
                batch = Transition(*zip(*batch))
                self.optimizer.zero_grad()
                s = self.compute_grad(batch)
                self.optimizer.step()
            print ('Finish replay in 20 iterations!')
        return stat
