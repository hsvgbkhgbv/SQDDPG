import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class CommNet(Agent):

    def __init__(self, args):
        '''
        args = {
        agent_num: int,
        hid_size: int,
        obs_size: int,
        continuous: bool,
        action_dim: int,
        comm_iters: int,
        action_heads_num: list(int)
        }
        '''
        super(CommNet, self).__init__()
        self.args = args

    def model(self):
        '''
        define the model of vanilla CommNet
        '''
        # encoder transforms observation to latent variables
        self.encoder = nn.Linear(self.args.obs_size, self.args.hid_size)
        # communication mask where the diagnal should be 0
        self.comm_mask = torch.ones(self.args.agent_num, self.args.agent_num) - torch.eye(self.args.agent_num, self.args.agent_num)
        # decoder transforms hidden variables to action vector
        if self.args.continuous:
            self.decoder = nn.Linear(self.args.hid_size, self.args.action_dim)
            self.action_log_std = nn.Parameter(torch.zeros(1, self.args.action_dim))
        else:
            self.action_heads = nn.ModuleList([nn.Linear(args.hid_size, o) for o in self.args.action_heads_num])
        # define communication inference
        self.f_module = nn.Linear(self.args.hid_size, self.args.hid_size)
        self.f_modules = nn.ModuleList([self.f_module for _ in range(self.args.comm_iters)])
        # define communication encoder
        self.C_module = nn.Linear(self.args.hid_size, self.args.hid_size)
        self.C_modules = nn.ModuleList([self.C_module for _ in range(self.args.comm_iters)])
        # initialise weights of communication encoder as 0
        for i in range(self.args.comm_iters):
            self.C_modules[i].weight.data.zero_()
        # define value function
        self.value_head = nn.Linear(self.hid_size, 1)

    def action(self, obs):
        '''
        define the action process of vanilla CommNet
        '''
        pass
