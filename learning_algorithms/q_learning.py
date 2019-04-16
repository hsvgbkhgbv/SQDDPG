from learning_algorithms.rl_algorithms import *
import torch
from utilities.util import *



class QLearning(ReinforcementLearning):

    def __init__(self, args):
        super(QLearning, self).__init__('Q_Learnning', args)
        assert self.args.q_func == True
        assert self.args.continuous == False
        assert self.args.target == True

    def __call__(self, batch, behaviour_net, target_net):
        return self.get_loss(batch, behaviour_net, target_net)

    def get_loss(self, batch, behaviour_net, target_net):
        batch_size = len(batch.state)
        n = self.args.agent_num
        action_dim = self.args.action_dim
        # collect the transition data
        rewards, last_step, done, actions, state, next_state = unpack_data(self.args, batch)
        # construct the computational graph
        action_out = behaviour_net.policy(state)
        values = behaviour_net.value(state, actions)
        values = torch.sum(values*actions, dim=-1)
        values = values.contiguous().view(-1, n)
        next_action_out = behaviour_net.policy(next_state)
        next_actions = select_action(self.args, next_action_out, status='test')
        next_values = behaviour_net.value(next_state, next_actions)
        next_values = torch.sum(next_values*next_actions, dim=-1)
        next_values = next_values.contiguous().view(-1, n)
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        # calculate the advantages
        assert values.size() == next_values.size()
        assert returns.size() == values.size()
        for i in range(rewards.size(0)):
            if last_step[i]:
                next_return = 0 if done[i] else next_values[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return.detach()
        deltas = returns - values
        # construct the action loss and the value loss
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        action_loss = 0
        return action_loss, value_loss, log_p_a
