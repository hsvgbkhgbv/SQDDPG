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
        self.action_optimizer = optim.Adam(self.behaviour_net.action_dict.parameters(), lr = args.policy_lrate)
        self.value_optimizer = optim.Adam(self.behaviour_net.value_dict.parameters(), lr = args.value_lrate)

    def get_episode(self, stat):
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
            action_out = self.behaviour_net.policy(state_, stat=stat)
            # return the sampled actions of all of agents
            action = select_action(self.args, action_out, status='train')
            # return the rescaled (clipped) actions
            _, actual = translate_action(self.args, action)
            # receive the reward and the next state
            next_state, reward, done, _ = self.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            # define the flag of the finish of exploration
            done = done or t == self.args.max_steps-1
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
        mean_reward = np.mean(mean_reward)
        num_steps = t+1
        return episode, mean_reward, num_steps

    def get_batch_results(self, batch):
        if self.args.training_strategy in ['ddpg']:
            action_loss, value_loss, log_p_a = self.rl(batch, self.behaviour_net, self.target_net)
        else:
            action_loss, value_loss, log_p_a = self.rl(batch, self.behaviour_net)
        return action_loss, value_loss, log_p_a

    def action_compute_grad(self, stat, batch_results):
        action_loss, log_p_a = batch_results
        if self.args.entr > 0:
            entropy = multinomial_entropy(log_p_a)
            action_loss -= self.args.entr * entropy
            stat['entropy'] = entropy.item()
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
        action_loss_ = 0
        value_loss_ = 0
        policy_grad_norm = 0
        value_grad_norm = 0
        for i in range(self.args.replay_iters):
            batch = self.replay_buffer.get_batch_episodes(\
                                    self.args.epoch_size*self.args.max_steps)
            batch = Transition(*zip(*batch))
            action_loss, value_loss, log_p_a = self.get_batch_results(batch)
            action_loss_ += action_loss.mean().item()
            value_loss_ += value_loss.mean().item()
            self.value_optimizer.zero_grad()
            self.value_compute_grad(value_loss)
            self.grad_clip(self.behaviour_net.value_dict)
            value_grad_norm += get_grad_norm(self.behaviour_net.value_dict)
            self.value_optimizer.step()
            self.action_optimizer.zero_grad()
            self.action_compute_grad(stat, (action_loss, log_p_a))
            self.grad_clip(self.behaviour_net.action_dict)
            policy_grad_norm += get_grad_norm(self.behaviour_net.action_dict)
            self.action_optimizer.step()
        stat['action_loss'] = action_loss_ / self.args.replay_iters
        stat['value_loss'] = value_loss_ / self.args.replay_iters
        stat['policy_grad_norm'] = policy_grad_norm / self.args.replay_iters
        stat['value_grad_norm'] = value_grad_norm / self.args.replay_iters

    def online_process(self, stat, batch):
        action_loss, value_loss, log_p_a = self.get_batch_results(batch)
        self.value_optimizer.zero_grad()
        self.value_compute_grad(value_loss)
        self.grad_clip(self.behaviour_net.value_dict)
        stat['value_grad_norm'] = get_grad_norm(self.behaviour_net.value_dict)
        self.value_optimizer.step()
        self.action_optimizer.zero_grad()
        self.action_compute_grad(stat, (action_loss, log_p_a))
        self.grad_clip(self.behaviour_net.action_dict)
        stat['policy_grad_norm'] = get_grad_norm(self.behaviour_net.action_dict)
        self.action_optimizer.step()
        stat['action_loss'] = action_loss.mean().item()
        stat['value_loss'] = value_loss.mean().item()

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
            if self.args.training_strategy in ['ddpg']:
                self.replay_buffer.add_experience(episode)
        stats['batch_finish_steps'] = len(batch)
        stats['mean_reward'] = average_mean_reward / self.args.epoch_size
        stats['average_episode_steps'] = average_num_steps / self.args.epoch_size
        batch = Transition(*zip(*batch))
        return batch, stats

    def train_batch(self, t, batch, stat):
        if self.args.training_strategy in ['ddpg']:
            self.replay_process(stat)
            if t%self.args.target_update_freq == self.args.target_update_freq - 1:
                params_target = list(self.target_net.parameters())
                params_behaviour = list(self.behaviour_net.parameters())
                for i in range(len(params_target)):
                    params_target[i] = (1 - self.args.target_lr) * params_target[i] + self.args.target_lr * params_behaviour[i]
                print ('traget net is updated!\n')
        else:
            self.online_process(stat, batch)
        return stat
