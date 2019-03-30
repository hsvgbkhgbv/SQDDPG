from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from util import *
from replay_buffer import *


# define a transition of an episode
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'start_step', 'last_step'))


class Trainer(object):

    def __init__(self, args, policy_net, env):
        self.args = args
        self.cuda = self.args.cuda and torch.cuda.is_available()
        self.policy_net = policy_net.cuda() if self.cuda else policy_net
        self.env = env
        self.optimizer = optim.RMSprop(policy_net.parameters(), lr = args.lrate, alpha=0.97, eps=1e-6)
        # self.optimizer = optim.SGD(policy_net.parameters(), lr = args.lrate, nesterov=True, momentum=0.1)
        self.params = [p for p in self.policy_net.parameters()]
        self.replay = self.args.replay
        if self.replay:
            self.replay_buffer = ReplayBuffer(int(1e7), 0.2)

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
            action_out, value = self.policy_net(prep_obs(state).contiguous().view(1, self.args.agent_num, self.args.obs_size))
            # return the sampled actions of all of agents
            action = select_action(self.args, action_out, 'train')
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
                trans = Transition(state, action.numpy(), np.array(reward), next_state, start_step, last_step)
                episode.append(trans)
                break
            else:
                last_step = False
                # record a transition
                trans = Transition(state, action.numpy(), np.array(reward), next_state, start_step, last_step)
                episode.append(trans)
            state = next_state
        mean_reward = np.array(mean_reward)
        mean_reward = mean_reward.mean()
        stat['num_steps'] = t + 1
        stat['mean_reward'] = mean_reward
        return (episode, stat)

    def compute_grad(self, batch):
        stat = dict()
        action_dim = self.args.action_dim
        n = self.args.agent_num
        batch_size = len(batch.state)
        # define some necessary containers
        rewards = torch.tensor(batch.reward, dtype=torch.float)
        last_step = torch.tensor(batch.last_step, dtype=torch.float).contiguous().view(-1, 1)
        start_step = torch.tensor(batch.start_step, dtype=torch.float).contiguous().view(-1, 1)
        if self.args.normalize_rewards:
            rewards = (rewards - rewards.mean(dim=0, keepdim=True)) / rewards.std(dim=0, keepdim=True)
        actions = list(zip(*batch.action))
        actions = torch.tensor(np.stack(actions[0], axis=0), dtype=torch.float32)
        if self.cuda:
            rewards = rewards.cuda()
        returns = torch.zeros((batch_size, n), dtype=torch.float)
        if self.args.training_strategy == 'actor_critic':
            deltas = torch.zeros((batch_size, n), dtype=torch.float)
            if self.cuda:
                deltas = deltas.cuda()
        advantages = torch.zeros((batch_size, n), dtype=torch.float)
        if self.cuda:
            returns = returns.cuda()
            advantages = advantages.cuda()
        # wrap the batch of states
        state = prep_obs(list(zip(batch.state)))
        next_state = prep_obs(list(zip(batch.next_state)))
        # construct the computational graph for the parameters
        action_out, values = self.policy_net(state)
        next_action_out, next_values = self.policy_net(next_state)
        if self.args.training_strategy == 'actor_critic':
            next_actions = select_action(self.args, next_action_out, 'train')
            next_values = next_values.gather(-1, next_actions.long())
            values = values.gather(-1, actions.long())
        values = values.contiguous().view(-1, n)
        next_values = next_values.contiguous().view(-1, n)
        if self.args.training_strategy == 'reinforce':
            assert returns.size() == rewards.size()
            # calculate the return reversely and the reward is shared
            for i in reversed(range(rewards.size(0))):
                if last_step[i]:
                    prev_coop_return = 0
                returns[i] = rewards[i] + self.args.gamma * prev_coop_return
                prev_coop_return = returns[i]
        elif self.args.training_strategy == 'actor_critic':
            assert rewards.size() == next_values.size()
            assert values.size() == next_values.size()
            # calculate the estimated action value
            for i in reversed(range(rewards.size(0))):
                if last_step[i]:
                    deltas[i] = rewards[i] - values[i]
                else:
                    deltas[i] = rewards[i] + self.args.gamma * next_values[i].detach() - values[i]
            deltas = deltas.contiguous().view(-1, 1)
            returns = deltas.detach()
        # calculate the advantage
        if self.args.training_strategy == 'reinforce':
            assert returns.size() == values.size()
            advantages = returns - values.detach()
        elif self.args.training_strategy == 'actor_critic':
            assert advantages.size() == returns.size()
            advantages = returns
        advantages = advantages.contiguous().view(-1, 1)
        # take the policy of actions
        if self.args.continuous:
            actions = actions.contiguous().view(-1, self.args.action_dim)
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = action_out
            log_prob = multinomials_log_density(actions, log_p_a).contiguous().view(-1, 1)
        # calculate the advantages
        assert log_prob.size() == advantages.size()
        action_loss = -advantages * log_prob
        action_loss = action_loss.sum() / batch_size
        stat['action_loss'] = action_loss.item()
        # calculate the value loss
        if self.args.training_strategy == 'reinforce':
            targets = returns
            value_loss = (targets - values).pow(2).view(-1)
        elif self.args.training_strategy == 'actor_critic':
            value_loss = deltas.pow(2).view(-1)
        value_loss = value_loss.sum() / batch_size
        stat['value_loss'] = value_loss.item()
        # combine the policy objective function and the value loss together
        loss = action_loss + self.args.value_coeff * value_loss
        # entropy regularization term, but it is obly available to discrete policy
        if not self.args.continuous:
            entropy = 0
            for i in range(log_p_a.size(0)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy
        # do the backpropogation
        loss.backward()
        return stat

    def run_batch(self):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while self.stats['num_episodes'] < self.args.epoch_size:
            episode, episode_stat = self.get_episode()
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
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
                                        self.args.epoch_size)
                batch = Transition(*zip(*batch))
                self.optimizer.zero_grad()
                s = self.compute_grad(batch)
                self.optimizer.step()
            print ('Finish replay in 20 iterations!')
        return stat
