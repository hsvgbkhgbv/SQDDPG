from learning_algorithms.rl_algorithms import *
from utilities.util import *



class DDPG(ReinforcementLearning):

    def __init__(self, args):
        super(DDPG, self).__init__('DDPG', args)

    def __call__(self, batch, behaviour_net, target_net):
        return self.get_loss(batch, behaviour_net, target_net)

    def get_loss(self, batch, behaviour_net, target_net):
        # TODO: fix policy params update
        batch_size = len(batch.state)
        n = self.args.agent_num
        # collect the transition data
        rewards, last_step, done, actions, state, next_state = behaviour_net.unpack_data(batch)
        # construct the computational graph
        # do the argmax action on the action loss
        action_out = behaviour_net.policy(state)
        actions_ = select_action(self.args, action_out, status='train', exploration=False)
        values_ = behaviour_net.value(state, actions_).contiguous().view(-1, n)
        # do the exploration action on the value loss
        values = behaviour_net.value(state, actions).contiguous().view(-1, n)
        # do the argmax action on the next value loss
        next_action_out = target_net.policy(next_state)
        next_actions_ = select_action(self.args, next_action_out, status='train', exploration=False)
        next_values_ = target_net.value(next_state, next_actions_.detach()).contiguous().view(-1, n)
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        assert values_.size() == next_values_.size()
        assert returns.size() == values.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_values_[i].detach()
            else:
                next_return = next_values_[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return
        deltas = returns - values
        advantages = values_
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        action_loss = -advantages
        action_loss = action_loss.mean(dim=0)
        value_loss = deltas.pow(2).mean(dim=0)
        return action_loss, value_loss, action_out
