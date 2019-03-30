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

    def __init__(self, args, policy_net, env, replay):
        self.args = args
        self.policy_net = policy_net.cuda() if torch.cuda.is_available() else policy_net
        self.env = env
        self.optimizer = optim.RMSprop(policy_net.parameters(), lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]
        self.replay = replay
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
            # print ('action_out')
            # print (action_out)
            # print ()
            # return the sampled actions of all of agents
            action = select_action(self.args, action_out, 'train')
            # print ('action')
            # print (action)
            # print ()
            # return the rescaled (clipped) actions
            _, actual = translate_action(self.args, action)
            # print ('actual')
            # print (actual)
            # print ()
            # receive the reward and the next state
            next_state, reward, done, _ = self.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            # define the flag of the finish of exploration
            done = done or t == self.args.max_steps - 1
            mean_reward.append(reward)
            # justify whether the game is done
            if done:
                last_step = True
                # record a transition
                trans = Transition(state, action, np.array(reward), next_state, start_step, last_step)
                episode.append(trans)
                break
            else:
                last_step = False
                # record a transition
                trans = Transition(state, action, np.array(reward), next_state, start_step, last_step)
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
        if self.args.normalize_rewards:
            rewards = (rewards - rewards.mean(dim=0, keepdim=True)) / rewards.std(dim=0, keepdim=True)
        actions = list(zip(*batch.action))
        # print (actions[0])
        actions = torch.stack(actions[0], dim=0).float()
        # print (actions)
        # actions = actions.transpose(1, 2).view(-1, n, 1)
        if torch.cuda.is_available():
            rewards = rewards.cuda()
        # action_out = list(zip(*batch.action_out))
        # action_out = [torch.cat(a, dim=0) for a in action_out]
        returns = torch.zeros((batch_size, n), dtype=torch.float)
        if self.args.training_strategy == 'actor_critic':
            deltas = torch.zeros((batch_size, n), dtype=torch.float)
            if torch.cuda.is_available():
                deltas = deltas.cuda()
        advantages = torch.zeros((batch_size, n), dtype=torch.float)
        if torch.cuda.is_available():
            returns = returns.cuda()
            advantages = advantages.cuda()
        # wrap the batch of states
        state = prep_obs(list(zip(batch.state)))
        next_state = prep_obs(list(zip(batch.next_state)))
        # print ('state')
        # print (state.shape)
        # construct the computational graph
        action_out, values = self.policy_net(state)
        # print ('grad')
        # print ('action_out')
        # print (action_out)
        # print ()
        next_action_out, next_values = self.policy_net(next_state)
        # action = select_action(self.args, action_out, 'train')
        # print ('action')
        # print (action)
        # print ()
        if self.args.training_strategy == 'actor_critic':
            next_actions = select_action(self.args, next_action_out, 'train')
            next_values = next_values.gather(-1, next_actions.long())
            # next_values = torch.cat([next_values[:, i, next_actions[:, i, 0]].unsqueeze(-1) for i in range(next_actions.size(1))], dim=-1)
            # values = torch.cat([values[:, i, actions[:, i, 0]].unsqueeze(-1) for i in range(action.size(1))], dim=-1)
            values = values.gather(-1, actions.long())
        values = values.contiguous().view(batch_size, n)
        next_values = next_values.contiguous().view(batch_size, n)
        # calculate the returns or estimated returns
        if self.args.training_strategy == 'reinforce':
            # calculate the return reversely and the reward is shared
            for i in reversed(range(rewards.size(0))):
                if batch.last_step:
                    prev_coop_return = 0
                returns[i] = rewards[i] + self.args.gamma * prev_coop_return
                prev_coop_return = returns[i]
        elif self.args.training_strategy == 'actor_critic':
            # calculate the estimated action value
            for i in range(rewards.size(0)):
                if batch.start_step:
                    I = 1
                deltas[i] = I * (rewards[i] + self.args.gamma * next_values[i].detach() - values[i])
                returns[i] = deltas[i]
                I *= self.args.gamma
        # calculate the advantage
        if self.args.training_strategy == 'reinforce':
            advantages = returns - values.data
        elif self.args.training_strategy == 'actor_critic':
            advantages = returns.data
        # take the policy of actions
        if self.args.continuous:
            actions = actions.contiguous().view(-1, self.args.action_dim)
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = action_out
            log_prob = multinomials_log_density(actions, log_p_a).squeeze(dim=-1)

        assert log_prob.size() == advantages.size()
        # calculate the advantages
        action_loss = -advantages * log_prob
        action_loss = action_loss.sum() / batch_size
        stat['action_loss'] = action_loss.item()
        # calculate the value loss
        if self.args.training_strategy == 'reinforce':
            targets = returns
            value_loss = (values - targets).pow(2).view(-1)
        elif self.args.training_strategy == 'actor_critic':
            value_loss = deltas.pow(2).view(-1)
        value_loss = value_loss.sum() / batch_size
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
