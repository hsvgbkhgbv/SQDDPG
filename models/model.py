import torch
import torch.nn as nn
import numpy as np
from utilities.util import *



class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.cuda_ = torch.cuda.is_available() and self.args.cuda
        self.n_ = self.args.agent_num
        self.hid_dim = self.args.hid_size
        self.obs_dim = self.args.obs_size
        self.act_dim = self.args.action_dim

    def reload_params_to_target(self):
        self.target_net.action_dict.load_state_dict( self.action_dict.state_dict() )
        self.target_net.value_dict.load_state_dict( self.value_dict.state_dict() )

    def update_target(self):
        for name, param in self.target_net.action_dict.state_dict().items():
            update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.action_dict.state_dict()[name]
            self.target_net.action_dict.state_dict()[name].copy_(update_params)
        for name, param in self.target_net.value_dict.state_dict().items():
            update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.value_dict.state_dict()[name]
            self.target_net.value_dict.state_dict()[name].copy_(update_params)

    def transition_update(self, trainer, trans, stat):
        if self.args.replay:
            trainer.replay_buffer.add_experience(trans)
            replay_cond = trainer.steps>self.args.replay_warmup\
             and len(trainer.replay_buffer.buffer)>=self.args.batch_size\
             and trainer.steps%self.args.behaviour_update_freq==self.args.behaviour_update_freq-1
            if replay_cond:
                trainer.action_replay_process(stat)
                for _ in range(self.args.critic_update_times):
                    trainer.value_replay_process(stat)
        else:
            trans_cond = trainer.steps%self.args.behaviour_update_freq==self.args.behaviour_update_freq-1
            if trans_cond:
                trainer.action_transition_process(stat, trans)
                for _ in range(self.args.critic_update_times):
                    trainer.value_replay_process(stat)
        if self.args.target:
            target_cond = trainer.steps%self.args.target_update_freq==self.args.target_update_freq-1
            if target_cond:
                self.update_target()

    def episode_update(self, trainer, episode, stat):
        if self.args.replay:
            trainer.replay_buffer.add_experience(episode)
            replay_cond = trainer.episodes>self.args.replay_warmup\
             and len(trainer.replay_buffer.buffer)>=self.args.batch_size\
             and trainer.episodes%self.args.behaviour_update_freq==self.args.behaviour_update_freq-1
            if replay_cond:
                trainer.action_replay_process(stat)
                for _ in range(self.args.critic_update_times):
                    trainer.value_replay_process(stat)
        else:
            episode = self.Transition(*zip(*episode))
            episode_cond = trainer.episodes%self.args.behaviour_update_freq==self.args.behaviour_update_freq-1
            if episode_cond:
                trainer.action_transition_process(stat)
                for _ in range(self.args.critic_update_times):
                    trainer.value_replay_process(stat)

    def construct_model(self):
        raise NotImplementedError()

    def get_agent_mask(self, batch_size, info):
        '''
        define the getter of agent mask to confirm the living agent
        '''
        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(self.n_)
            num_agents_alive = self.n_
        # shape = (1, 1, n)
        agent_mask = agent_mask.view(1, 1, self.n_)
        # shape = (batch_size, n ,n, 1)
        agent_mask = cuda_wrapper(agent_mask.expand(batch_size, self.n_, self.n_).unsqueeze(-1), self.cuda_)
        return num_agents_alive, agent_mask

    def policy(self, obs, last_act=None, last_hid=None, gate=None, info={}, stat={}):
        raise NotImplementedError()

    def value(self, obs, act):
        raise NotImplementedError()

    def construct_policy_net(self):
        raise NotImplementedError()

    def construct_value_net(self):
        raise NotImplementedError()

    def init_weights(self, m):
        '''
        initialize the weights of parameters
        '''
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.args.init_std)

    def get_loss(self):
        raise NotImplementedError()
