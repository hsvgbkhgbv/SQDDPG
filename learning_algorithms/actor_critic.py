from learning_algorithms.rl_algorithms import *
import torch
from utilities.util import *



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
        rewards, last_step, done, actions, state, next_state = unpack_data(self.args, batch)
        # construct the computational graph
        action_out = behaviour_net.policy(state)
        values = behaviour_net.value(state, actions)
        if self.args.q_func:
            values = torch.sum(values*actions, dim=-1)
        values = values.contiguous().view(-1, n)
        next_action_out = behaviour_net.policy(next_state)
        next_actions = select_action(self.args, next_action_out, status='train')
        next_values = behaviour_net.value(next_state, next_actions)
        if self.args.q_func:
            next_values = torch.sum(next_values*next_actions, dim=-1)
        next_values = next_values.contiguous().view(-1, n)
        # calculate the advantages
        assert values.size() == next_values.size()
        deltas = rewards + self.args.gamma * next_values.detach() - values
        advantages = values.detach()
        # construct the action loss and the value loss
        if self.args.continuous:
            action_means = actions.contiguous().view(-1, self.args.action_dim)
            action_stds = cuda_wrapper(torch.ones_like(action_means), self.cuda_)
            log_p_a = normal_log_density(actions.detach(), action_means, action_stds)
            log_prob = log_p_a.clone()
        else:
            log_p_a = action_out
            log_prob = multinomials_log_density(actions, log_p_a).contiguous().view(-1, 1)
        advantages = advantages.contiguous().view(-1, 1)
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        assert log_prob.size() == advantages.size()
        action_loss = -advantages * log_prob
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, log_p_a
