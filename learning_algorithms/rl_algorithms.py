import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utilities.util import *



class ReinforcementLearning(object):

    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.cuda_ = self.args.cuda
        print ( '{}\n'.format(args) )

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
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        state = cuda_wrapper(prep_obs(list(zip(batch.state))), self.cuda_)
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda_)
        return (rewards, last_step, start_step, actions, returns, state, next_state)
