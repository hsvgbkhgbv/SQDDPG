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
        self.cuda = self.args.cuda
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
        rewards = cuda_wrapper(torch.tensor(batch.reward, dtype=torch.float), self.cuda)
        last_step = cuda_wrapper(torch.tensor(batch.last_step, dtype=torch.float).contiguous().view(-1, 1), self.cuda)
        start_step = cuda_wrapper(torch.tensor(batch.start_step, dtype=torch.float).contiguous().view(-1, 1), self.cuda)
        actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.action))[0], axis=0), dtype=torch.float), self.cuda)
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda)
        state = cuda_wrapper(prep_obs(list(zip(batch.state))), self.cuda)
        return (rewards, last_step, start_step, actions, returns, state)



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
        rewards, last_step, start_step, actions, returns, state = self.unpack_data(batch)
        # construct the computational graph
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda)
        action_out = behaviour_net.policy(state)
        # TODO: How to construct the backprop at this node for ddpg when the action is discrete
        values = behaviour_net.value(actions).contiguous().view(-1, n)
        # calculate the return
        assert returns.size() == rewards.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                prev_coop_return = 0
            returns[i] = rewards[i] + self.args.gamma * prev_coop_return
            prev_coop_return = returns[i]
        # construct the action loss and the value loss
        deltas = returns - values
        advantages = deltas.contiguous().view(-1, 1).detach()
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        if self.args.continuous:
            actions = actions.contiguous().view(-1, self.args.action_dim)
            action_means, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_stds)
        else:
            log_p_a = action_out
            log_prob = multinomials_log_density(actions, log_p_a).contiguous().view(-1, 1)
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
        rewards, last_step, start_step, actions, returns, state = self.unpack_data(batch)
        # construct the computational graph
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda)
        action_out = behaviour_net.policy(state)
        # TODO: How to construct the backprop at this node for ddpg when the action is discrete
        values = behaviour_net.value(actions).contiguous().view(-1, n)
        next_action_out = behaviour_net.policy(next_state)
        # TODO: How to construct the backprop at this node for ddpg when the action is discrete
        next_actions = select_action(self.args, next_action_out, 'train')
        next_values = behaviour_net.value(next_actions).contiguous().view(-1, n)
        # calculate the advantages
        deltas = cuda_wrapper(torch.zeros_like(values), self.cuda)
        assert values.size() == next_values.size()
        for i in range(rewards.size(0)):
            if last_step[i]:
                deltas[i] = rewards[i] - values[i]
            else:
                deltas[i] = rewards[i] + self.args.gamma * next_values[i].detach() - values[i]
        # construct the action loss and the value loss
        advantages = deltas.detach()
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        if self.args.continuous:
            actions = actions.contiguous().view(-1, self.args.action_dim)
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = action_out
            log_prob = multinomials_log_density(actions, log_p_a).contiguous().view(-1, 1)
        advantages = advantages.contiguous().view(-1, 1)
        assert log_prob.size() == advantages.size()
        action_loss = -advantages * log_prob
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, log_p_a



class DDPG(ReinforcementLearning):

    def __init__(self, args):
        super(ActorCritic, self).__init__('DDPG', args)

    def __call__(self, batch, behaviour_net, target_net):
        return self.get_loss(batch, behaviour_net, target_net)

    def get_loss(self, batch, behaviour_net, target_net):
        batch_size = len(batch.state)
        n = self.args.agent_num
        action_dim = self.args.action_dim
        # collect the transition data
        rewards, last_step, start_step, actions, returns, state = self.unpack_data(batch)
        # construct the computational graph
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda)
        action_out, values = behaviour_net(state)
        values = values.contiguous().view(-1, n)
        next_action_out, next_values = target_net(next_state)
        next_values = next_values.contiguous().view(-1, n)
        if self.args.continuous:
            action_tensor = cuda_wrapper(torch.zeros(tuple(actions.size()[:-1])+(self.args.action_dim,)))
            action_tensor.scatter_(-1, actions.long(), 1)
            action_means, action_log_stds, action_stds = action_out
        else:

            tran_actions = action_tensor
            assert tran_actions.size() == action_out.size()
            assert action_out.size() == values.size()
            values_ = (torch.ceil(action_out * tran_actions) * values).sum(dim=-1)
            values = values.gather(-1, actions.long())
            next_actions = select_action(self.args, next_action_out, 'train')
            next_values = next_values.gather(-1, next_actions.long())
