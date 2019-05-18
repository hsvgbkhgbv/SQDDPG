from learning_algorithms.rl_algorithms import *
from utilities.util import *



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
        rewards, last_step, done, actions, state, next_state = behaviour_net.unpack_data(batch)
        # construct the computational graph
        action_out = behaviour_net.policy(state)
        values = behaviour_net.value(state).contiguous().view(-1, n)
        # get the next actions and the next values
        next_action_out = behaviour_net.policy(next_state)
        next_actions = select_action(self.args, next_action_out.detach(), status='train')
        next_values = behaviour_net.value(next_state).contiguous().view(-1, n)
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        # calculate the return
        assert returns.size() == rewards.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_values[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return
            next_return = returns[i]
        # construct the action loss and the value loss
        deltas = returns - values
        advantages = deltas.contiguous().view(-1, 1).detach()
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        if self.args.continuous:
            action_means = actions.contiguous().view(-1, self.args.action_dim)
            action_stds = cuda_wrapper(torch.ones_like(action_means), self.cuda_)
            log_prob_a = normal_log_density(actions.detach(), action_means, action_stds)
        else:
            log_prob_a = multinomials_log_density(actions.detach(), action_out).contiguous().view(-1, 1)
        assert log_prob.size() == advantages.size()
        action_loss = -advantages * log_prob
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, log_prob_a
