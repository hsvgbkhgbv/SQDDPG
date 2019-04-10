import numpy as np
import torch
import numbers
import math
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.normal import Normal



class GumbelSoftmax(OneHotCategorical):

    def __init__(self, logits, temperature=0.1):
        super(GumbelSoftmax, self).__init__(logits=logits)
        self.eps = 1e-20
        self.temperature = temperature

    def sample_gumbel(self):
        U = self.logits.clone()
        U.uniform_(0, 1.0)
        return -torch.log( -torch.log( U + self.eps ) )

    def gumbel_softmax_sample(self):
        y = self.logits + self.sample_gumbel()
        return torch.softmax( y / self.temperature, dim=-1)

    def hard_gumbel_softmax(self):
        y = self.gumbel_softmax_sample()
        return (torch.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample()

    def sample(self):
        return self.rsample().detach()


def normal_entropy(mean, std):
    return Normal(mean, std).entropy().sum()

def multinomial_entropy(log_probs):
    assert log_probs.size(-1) > 1
    return GumbelSoftmax(logits=log_probs).entropy().sum()

def normal_log_density(x, mean, std):
    return Normal(mean, std).log_prob(x)

def multinomials_log_density(actions, log_probs):
    assert log_probs.size(-1) > 1
    return GumbelSoftmax(logits=log_probs).log_prob(actions)

def select_action(args, action_out, status='train', exploration=True):
    if args.continuous:
        act_mean = action_out
        act_std = cuda_wrapper(torch.ones_like(act_mean), args.cuda)
        if status == 'train':
            return Normal(act_mean, act_std).rsample()
        elif status == 'test':
            return act_mean
    else:
        log_p_a = action_out
        if status == 'train':
            if exploration:
                if args.training_strategy in ['ddpg']:
                    return GumbelSoftmax(logits=log_p_a).sample()
                else:
                    return OneHotCategorical(logits=log_p_a).sample()
            else:
                assert args.training_strategy in ['ddpg']
                return GumbelSoftmax(logits=log_p_a).rsample()
        elif status == 'test':
            p_a = torch.softmax(log_p_a, dim=-1)
            return  (p_a == torch.max(p_a, dim=-1, keepdim=True)[0]).float()

def translate_action(args, action):
    if args.action_num > 1:
        actual = [act.detach().squeeze().cpu().numpy() for act in torch.unbind(action, 1)]
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

def prep_obs(state=[]):
    state = np.array(state)
    if len(state.shape) == 2:
        state = np.stack(state, axis=0)
    elif len(state.shape) == 4:
        state = np.concatenate(state, axis=0)
    else:
        raise RuntimeError('The shape of the observation is incorrect.')
    return torch.tensor(state).float()

def cuda_wrapper(tensor, cuda):
    if isinstance(tensor, torch.Tensor):
        return tensor.cuda() if cuda else tensor
    else:
        raise RuntimeError('Please enter a pytorch tensor, now a {} is received.'.format(type(tensor)))

def batchnorm(batch):
    if isinstance(batch, torch.Tensor):
        assert batch.size(-1) == 1
        return (batch - batch.mean()) / batch.std()
    else:
        raise RuntimeError('Please enter a pytorch tensor, now a {} is received.'.format(type(batch)))

def get_grad_norm(module):
    grad_norms = []
    for name, param in module.named_parameters():
        grad_norms.append(torch.norm(param.grad))
    return np.mean(grad_norms)
