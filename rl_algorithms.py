import numpy as np
import torch
from torch import optim
import torch.nn as nn
from util import *
from arguments import *



class ReinforcementLearning(object):

    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.cuda_ = self.args.cuda
        print (args)

    def __str__(self):
        print (self.name)

    def __call__(self):
        raise NotImplementedError()

    def get_loss(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def unpack_data(self, batch):
        batch_size = len(batch.state)
        n = self.args.agent_num
        action_dim = self.args.action_dim
        rewards = cuda_wrapper(torch.tensor(batch.reward, dtype=torch.float), self.cuda_)
        last_step = cuda_wrapper(torch.tensor(batch.last_step, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        start_step = cuda_wrapper(torch.tensor(batch.start_step, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.action))[0], axis=0), dtype=torch.float), self.cuda_)
        # actions = cuda_wrapper(torch.stack(list(zip(*batch.action))[0], dim=0), self.cuda_)
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        state = cuda_wrapper(prep_obs(list(zip(batch.state))), self.cuda_)
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda_)
        return (rewards, last_step, start_step, actions, returns, state, next_state)



class REINFORCE(ReinforcementLearning):

    def __init__(self, args):
        super(REINFORCE , self).__init__('REINFORCE', args)

    def __call__(self, batch, behaviour_net):
        return self.get_loss(batch, behaviour_net)

    def get_loss(self, batch, behaviour_net):
        batch_size = len(batch.state)
        n = self.args.agent_num
        action_dim = self.args.action_dim
        # collect the transition data
        rewards, last_step, start_step, actions, returns, state, next_state = self.unpack_data(batch)
        # construct the computational graph
        action_out = behaviour_net.policy(state)
        # TODO: How to construct the backprop at this node for ddpg when the action is discrete
        values = behaviour_net.value(state, actions.detach()).contiguous().view(-1, n)
        # get the next actions and the next values
        next_action_out = behaviour_net.policy(next_state)
        next_actions = select_action(self.args, next_action_out.detach(), status='train')
        next_values = behaviour_net.value(next_state, next_actions.detach()).contiguous().view(-1, n)
        # calculate the return
        assert returns.size() == rewards.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                prev_coop_return = next_values[i].detach()
            returns[i] = rewards[i] + self.args.gamma * prev_coop_return
            prev_coop_return = returns[i]
        # construct the action loss and the value loss
        deltas = returns - values
        advantages = deltas.contiguous().view(-1, 1).detach()
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        if self.args.continuous:
            action_means = actions.contiguous().view(-1, self.args.action_dim)
            action_stds = cuda_wrapper(torch.ones_like(action_means), self.cuda_)
            log_prob = normal_log_density(actions.detach(), action_means, action_stds)
        else:
            log_p_a = action_out
            log_prob = multinomials_log_density(actions.detach(), log_p_a).contiguous().view(-1, 1)
        assert log_prob.size() == advantages.size()
        action_loss = -advantages * log_prob
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, log_p_a



class ActorCritic(ReinforcementLearning):

    def __init__(self, args):
        super(ActorCritic, self).__init__('Actor_Critic', args)

    def __call__(self, batch, behaviour_net):
        return self.get_loss(batch, behaviour_net)

    def get_loss(self, batch, behaviour_net):
        batch_size = len(batch.state)
        n = self.args.agent_num
        action_dim = self.args.action_dim
        # collect the transition data
        rewards, last_step, start_step, actions, returns, state, next_state = self.unpack_data(batch)
        # construct the computational graph
        action_out = behaviour_net.policy(state)
        # TODO: How to construct the backprop at this node for ddpg when the action is discrete
        values = behaviour_net.value(state, actions.detach()).contiguous().view(-1, n)
        next_action_out = behaviour_net.policy(next_state)
        # TODO: How to construct the backprop at this node for ddpg when the action is discrete
        next_actions = select_action(self.args, next_action_out.detach(), status='train')
        next_values = behaviour_net.value(next_state, next_actions).contiguous().view(-1, n)
        # calculate the advantages
        deltas = cuda_wrapper(torch.zeros_like(values), self.cuda_)
        assert values.size() == next_values.size()
        deltas = rewards + self.args.gamma * next_values.detach() - values
        advantages = deltas.detach()
        # construct the action loss and the value loss
        if self.args.continuous:
            action_means = actions.contiguous().view(-1, self.args.action_dim)
            action_stds = cuda_wrapper(torch.ones_like(action_means), self.cuda_)
            log_prob = normal_log_density(actions.detach(), action_means, action_stds)
        else:
            log_p_a = action_out
            log_prob = multinomials_log_density(actions.detach(), log_p_a).contiguous().view(-1, 1)
        advantages = advantages.contiguous().view(-1, 1)
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        assert log_prob.size() == advantages.size()
        action_loss = -advantages * log_prob
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, log_p_a



class DDPG(ReinforcementLearning):

    def __init__(self, args):
        super(DDPG, self).__init__('DDPG', args)

    def __call__(self, batch, behaviour_net, target_net):
        return self.get_loss(batch, behaviour_net, target_net)

    def get_loss(self, batch, behaviour_net, target_net):
        batch_size = len(batch.state)
        n = self.args.agent_num
        action_dim = self.args.action_dim
        # collect the transition data
        rewards, last_step, start_step, actions, returns, state, next_state = self.unpack_data(batch)
        # construct the computational graph
        # do the argmax action on the action loss
        action_out = behaviour_net.policy(state)
        # actions_ = select_action(args, action_out, status='train', exploration=False)
        actions_ = torch.softmax(action_out, dim=-1)
        values_ = behaviour_net.value(state, actions_).contiguous().view(-1, n)
        # do the exploration action on the value loss
        values = behaviour_net.value(state, actions).contiguous().view(-1, n)
        # do the argmax action on the next value loss
        next_action_out = target_net.policy(next_state)
        # next_actions = select_action(self.args, next_action_out, status='train', exploration=False)
        next_actions = torch.softmax(next_action_out, dim=-1)
        next_values_ = target_net.value(next_state, next_actions.detach()).contiguous().view(-1, n)
        assert values_.size() == next_values_.size()
        deltas = rewards + self.args.gamma * next_values_.detach() - values
        advantages = values_
        advantages = advantages.contiguous().view(-1, 1)
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        if self.args.continuous:
            action_means = actions.contiguous().view(-1, self.args.action_dim)
            action_stds = cuda_wrapper(torch.ones_like(action_means), self.cuda_)
            log_prob = normal_log_density(actions.detach(), action_means, action_stds)
        else:
            log_p_a = action_out
            log_prob = multinomials_log_density(actions.detach(), log_p_a).contiguous().view(-1, 1)
        action_loss = -advantages
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, log_p_a
