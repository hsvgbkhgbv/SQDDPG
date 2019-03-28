import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IC3Net(nn.Module):

    def __init__(self, args):
        '''
        args = (
        agent_num: int,
        hid_size: int,
        obs_size: [int],
        continuous: bool,
        action_dim: [int],
        comm_iters: int,
        init_std: float,
        'lrate': float,
        'batch_size': int,
        'max_steps': int,
        'gamma': float,
        'normalize_rewards': bool,
        'advantages_per_action': bool,
        'value_coeff': float,
        'entr': float,
        'action_num'=int,
        'skip_connection'=bool
        )
        args is a namedtuple, e.g. args = collections.namedtuple()
        '''
        super(IC3Net, self).__init__()
        self.args = args
        # create a model
        self.construct_model()
        # initialize parameters with normal distribution with mean of 0
        # map(self.init_weights, self.parameters())
        self.apply(self.init_weights)

    def mask_obs(self, x):
        x_lens = [len(x_) for x_ in x]
        x_len_max = np.max(x_lens)
        for i in range(len(x_lens)):
            if x_lens[i] < x_len_max:
                x[i] = np.concatenate((x[i], np.zeros(x_len_max-x_lens[i])), axis=0)
        return torch.Tensor(x).float().unsqueeze(0).cuda() if torch.cuda.is_available() else torch.Tensor(x).float().unsqueeze(0)

    def construct_model(self):
        '''
        define the model of vanilla CommNet
        '''
        # encoder transforms observation to latent variables
        self.encoder = nn.Linear(self.args.obs_size, self.args.hid_size)
        # communication mask where the diagnal should be 0
        self.comm_mask = torch.ones(self.args.agent_num, self.args.agent_num) - torch.eye(self.args.agent_num, self.args.agent_num)
        if torch.cuda.is_available():
            self.comm_mask = self.comm_mask.cuda()
        # decoder transforms hidden states to action vector
        if self.args.continuous:
            self.action_mean = nn.Linear(self.args.hid_size, self.args.action_dim)
            self.action_log_std = nn.Parameter(torch.zeros(1, self.args.action_dim))
        else:
            self.action_head = nn.Linear(self.args.hid_size, self.args.action_dim)
        # define communication inference
        self.f_module = nn.LSTMCell(self.args.hid_size, self.args.hid_size)
        # define the gate function
        self.g_module = nn.Linear(self.args.hid_size, self.args.agent_num)
        self.g_modules = nn.ModuleList([self.g_module for _ in range(self.args.comm_iters)])
        if self.args.training_strategy == 'reinforce':
            # define value function
            self.value_head = nn.Linear(self.args.hid_size, 1)
        elif self.args.training_strategy == 'actor_critic':
            # define action value function
            self.action_value_head = nn.Linear(self.args.hid_size, self.args.action_dim)
        self.tanh = nn.Tanh()

    def state_encoder(self, x):
        '''
        define a state encoder
        '''
        return self.tanh(self.encoder(x))

    def get_agent_mask(self, batch_size, info):
        '''
        define the getter of agent mask to confirm the living agent
        '''
        n = self.args.agent_num
        with torch.no_grad():
            if 'alive_mask' in info:
                agent_mask = torch.from_numpy(info['alive_mask'])
                num_agents_alive = agent_mask.sum()
            else:
                agent_mask = torch.ones(n)
                num_agents_alive = n
        # shape = (1, 1, n)
        agent_mask = agent_mask.view(1, 1, n)
        # shape = (batch_size, n ,n, 1)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1)
        if torch.cuda.is_available():
            agent_mask = agent_mask.cuda()
        return num_agents_alive, agent_mask

    def action(self, obs, info={}):
        '''
        define the action process of vanilla CommNet
        '''
        with torch.no_grad():
            obs = self.mask_obs(obs)
        # encode observation
        e = self.state_encoder(obs)
        # get the batch_size \times
        batch_size = obs.size(0)
        # get the initial state
        h, cell = self.init_hidden(batch_size)
        # get the total number of agents including dead
        n = self.args.agent_num
        # get the agent mask
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        # conduct the main process of communication
        for i in range(self.args.comm_iters):
            h_ = h.view(batch_size, n, self.args.hid_size)
            # define the gate function
            gate_ = torch.sigmoid(self.g_modules[i](h_))
            gate = torch.ceil(gate_)
            # shape = (batch_size, n, hid_size)->(batch_size, n, 1, hid_size)->(batch_size, n, n, hid_size)
            h_ = h_.unsqueeze(-2).expand(-1, n, n, self.args.hid_size)
            # construct the communication mask
            mask = self.comm_mask.view(1, n, n) # shape = (1, n, n)
            mask = mask.expand(batch_size, n, n) # shape = (batch_size, n, n)
            mask = mask.unsqueeze(-1) # shape = (batch_size, n, n, 1)
            mask = mask.expand_as(h_) # shape = (batch_size, n, n, hid_size)
            # construct the commnication gate
            gate = gate.view(1, n, n) # shape = (1, n, n)
            gate = gate.expand(batch_size, n, n) # shape = (batch_size, n, n)
            gate = gate.unsqueeze(-1) # shape = (batch_size, n, n, 1)
            gate = gate.expand_as(h_) # shape = (batch_size, n, n, hid_size)
            # mask each agent itself (collect the hidden state of other agents)
            h_ = h_ * gate * mask
            # mask the dead agent
            h_ = h_ * agent_mask * agent_mask.transpose(1, 2)
            # average the hidden state
            h_ = h_ / (num_agents_alive - 1)
            # calculate the communication vector
            c = h_.sum(dim=1) if i != 0 else torch.zeros_like(h) # shape = (batch_size, n, hid_size)
            inp = e+c
            inp = inp.view(batch_size*n, self.args.hid_size)
            # f_moudle
            h, cell = self.f_module(inp, (h, cell))
        h = h.view(batch_size, n, self.args.hid_size)
        if self.args.training_strategy == 'reinforce':
            # calculate the value function (baseline)
            value_head = self.value_head(h)
        elif self.args.training_strategy == 'actor_critic':
            if self.args.continuous:
                value_head = self.value_head(h)
            else:
                value_head = self.action_value_head(h)
        # calculate the action vector (policy)
        if self.args.continuous:
            # shape = (batch_size, n, action_dim)
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions, shape = (batch_size, n, action_type, action_num)
            action = F.log_softmax(self.action_head(h), dim=-1)
        return action, value_head

    def forward(self, obs, info={}):
        return self.action(obs, info)

    def init_weights(self, m):
        '''
        initialize the weights of parameters
        '''
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.args.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple((torch.zeros(batch_size * self.args.agent_num, self.args.hid_size),
                      torch.zeros(batch_size * self.args.agent_num, self.args.hid_size)))
