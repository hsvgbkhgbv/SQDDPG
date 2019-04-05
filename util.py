import numpy as np
import torch
import numbers
import math



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

def normal_entropy(mean, std):
    return torch.distributions.normal.Normal(mean, std).entropy()

def multinomial_entropy(log_probs):
    assert log_probs.size(-1) > 1
    return torch.distributions.one_hot_categorical.OneHotCategorical(logits=log_probs).entropy().sum()

def normal_log_density(x, mean, std):
    return torch.distributions.normal.Normal(mean, std).log_prob(x)

def multinomials_log_density(actions, log_probs):
    return torch.distributions.one_hot_categorical.OneHotCategorical(logits=log_probs).log_prob(actions)

def select_action(args, action_out, status='train'):
    if args.continuous:
        act_mean, act_std = action_out
        if status == 'train':
            return torch.distributions.normal.Normal(act_mean, act_std).sample()
        elif status == 'test':
            return act_mean
    else:
        log_p_a = action_out
        if status == 'train':
            return torch.distributions.one_hot_categorical.OneHotCategorical(logits=log_p_a).sample()
        elif status == 'test':
            return torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(1e-35, logits=log_p_a).sample()

def translate_action(args, action):
    if args.action_num > 1:
        actual = [act.squeeze().numpy() for act in torch.unbind(action, 1)]
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
        if cuda:
            return tensor.cuda()
        else:
            return tensor
    else:
        raise RuntimeError('Please enter a pytorch tensor, now a {} is received.'.format(type(tensor)))

def batchnorm(batch):
    if isinstance(batch, torch.Tensor):
        batch_norm = (batch - batch.mean()) / batch.std()
        return batch_norm
    else:
        raise RuntimeError('Please enter a pytorch tensor, now a {} is received.'.format(type(batch)))
