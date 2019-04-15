from learning_algorithms.rl_algorithms import *
from utilities.util import *



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
        rewards, last_step, done, actions, state, next_state = unpack_data(self.args, batch)
        # construct the computational graph
        # do the argmax action on the action loss
        action_out = behaviour_net.policy(state)
        actions_ = select_action(self.args, action_out, status='train', exploration=False)
        # actions_ = torch.softmax(action_out, dim=-1)
        values_ = behaviour_net.value(state, actions_).contiguous().view(-1, n)
        # do the exploration action on the value loss
        values = behaviour_net.value(state, actions).contiguous().view(-1, n)
        # do the argmax action on the next value loss
        next_action_out = target_net.policy(next_state)
        next_actions = select_action(self.args, next_action_out, status='train', exploration=False)
        # next_actions = torch.softmax(next_action_out, dim=-1)
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
            log_p_a = normal_log_density(actions.detach(), action_means, action_stds)
            log_p_prob = log_p_a.clone()
        else:
            log_p_a = action_out
            log_prob = multinomials_log_density(actions.detach(), log_p_a).contiguous().view(-1, 1)
        action_loss = -advantages
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, log_p_a
