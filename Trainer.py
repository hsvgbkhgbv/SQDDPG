from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn


def merge_stat(src, dest):
    for k, v in src.items():
        if not k in dest:
            dest[k] = v
        elif isinstance(v, numbers.Number):
            dest[k] = dest.get(k, 0) + v
        elif isinstance(v, np.ndarray): # for rewards in case of multi-agent
            dest[k] = dest.get(k, 0) + v
        else:
            if isinstance(dest[k], list) and isinstance(v, list):
                dest[k].extend(v)
            elif isinstance(dest[k], list):
                dest[k].append(v)
            else:
                dest[k] = [dest[k], v]

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def multinomials_log_density(actions, log_probs):
    log_prob = 0
    for i in range(len(log_probs)):
        log_prob += log_probs[i].gather(1, actions[:, i].long().unsqueeze(1))
    return log_prob

def multinomials_log_densities(actions, log_probs):
    log_prob = [0] * len(log_probs)
    for i in range(len(log_probs)):
        log_prob[i] += log_probs[i].gather(1, actions[:, i].long().unsqueeze(1))
    log_prob = torch.cat(log_prob, dim=-1)
    return log_prob

def select_action(args, action_out):
    if args.continuous:
        action_mean, _, action_std = action_out
        action = torch.normal(action_mean, action_std)
        return action.detach()
    else:
        log_p_a = action_out
        p_a = [[z.exp() for z in x] for x in log_p_a]
        ret = torch.stack([torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_a])
        return ret

def translate_action(args, env, action):
    if args.action_num[0] > 0:
        # environment takes discrete action
        action = [x.squeeze().data.numpy() for x in action]
        actual = action
        return action, actual
    else:
        if args.continuous:
            action = action.data[0].numpy()
            cp_action = action.copy()
            # clip and scale action to correct range
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                cp_action[i] = cp_action[i] * args.action_scale
                cp_action[i] = max(-1.0, min(cp_action[i], 1.0))
                cp_action[i] = 0.5 * (cp_action[i] + 1.0) * (high - low) + low
            return action, cp_action
        else:
            actual = np.zeros(len(action))
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                actual[i] = action[i].data.squeeze()[0] * (high - low) / (args.naction_heads[i] - 1) + low
            action = [x.squeeze().data[0] for x in action]
            return action, actual


# define a transition of an episode
Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state', 'reward', 'misc'))


class Trainer(object):

    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.optimizer = optim.RMSprop(policy_net.parameters(), lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]

    def get_episode(self):
        # define the episode list
        episode = []
        # reset the environment
        state = self.env.reset()
        # set up two auxilliary dictionaries
        stat = dict()
        info = dict()
        # define the main process of exploration
        for t in range(self.args.max_steps):
            misc = dict()
            # decide the next action and return the correlated state value (baseline)
            action_out, value = self.policy_net.action(state, info)
            # return the sampled actions of all of agents
            action = select_action(self.args, action_out)
            # return the rescaled (clipped) actions
            _, actual = translate_action(self.args, self.env, action[0])
            # receive the reward and the next state
            action_wrapper = []
            for a in actual[0]:
                action_ = np.zeros(5)
                action_[a] += 1
                action_wrapper.append(np.concatenate([action_, np.zeros(self.env.world.dim_c)]))
            next_state, reward, done, info = self.env.step(action_wrapper)
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

            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']
        return (episode, stat)

    def compute_grad(self, batch):

        stat = dict()

        action_num = self.args.action_num
        action_dim = self.args.action_dim
        n = self.args.agent_num
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward)
        episode_masks = torch.Tensor(batch.episode_mask)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)

        coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
        deltas = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            # calculate the return reversely and the reward is shared
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                         + ((1 - self.args.mean_ratio) * ncoop_returns[i])

        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)
            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
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
            # merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    def train_batch(self):
        batch, stat = self.run_batch()
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat
