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
        self.target_net.action_dicts.load_state_dict( self.action_dicts.state_dict() )
        self.target_net.value_dicts.load_state_dict( self.value_dicts.state_dict() )

    def update_target(self):
        for name, param in self.target_net.action_dicts.state_dict().items():
            update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.action_dicts.state_dict()[name]
            self.target_net.action_dicts.state_dict()[name].copy_(update_params)
        for name, param in self.target_net.value_dicts.state_dict().items():
            update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.value_dicts.state_dict()[name]
            self.target_net.value_dicts.state_dict()[name].copy_(update_params)

    def transition_update(self, trainer, trans, stat):
        if self.args.replay:
            trainer.replay_buffer.add_experience(trans)
            replay_cond = trainer.steps>self.args.replay_warmup\
             and len(trainer.replay_buffer.buffer)>=self.args.batch_size\
             and trainer.steps%self.args.behaviour_update_freq==0
            if replay_cond:
                for _ in range(self.args.critic_update_times):
                    trainer.value_replay_process(stat)
                trainer.action_replay_process(stat)
                # TODO: hard code
                # clear replay buffer for on policy algorithm
                if self.__class__.__name__ in ["COMAFC","MFAC","IndependentAC"] :
                    trainer.replay_buffer.clear()
        else:
            trans_cond = trainer.steps%self.args.behaviour_update_freq==0
            if trans_cond:
                for _ in range(self.args.critic_update_times):
                    trainer.value_replay_process(stat)
                trainer.action_transition_process(stat, trans)
        if self.args.target:
            target_cond = trainer.steps%self.args.target_update_freq==0
            if target_cond:
                self.update_target()

    def episode_update(self, trainer, episode, stat):
        if self.args.replay:
            trainer.replay_buffer.add_experience(episode)
            replay_cond = trainer.episodes>self.args.replay_warmup\
             and len(trainer.replay_buffer.buffer)>=self.args.batch_size\
             and trainer.episodes%self.args.behaviour_update_freq==0
            if replay_cond:
                for _ in range(self.args.critic_update_times):
                    trainer.value_replay_process(stat)
                trainer.action_replay_process(stat)
        else:
            episode = self.Transition(*zip(*episode))
            episode_cond = trainer.episodes%self.args.behaviour_update_freq==0
            if episode_cond:
                for _ in range(self.args.critic_update_times):
                    trainer.value_replay_process(stat)
                trainer.action_transition_process(stat)

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

    def credit_assignment_demo(self, obs, act):
        assert isinstance(obs, np.ndarray)
        assert isinstance(act, np.ndarray)
        obs = cuda_wrapper(torch.tensor(obs).float(), self.cuda_)
        act = cuda_wrapper(torch.tensor(act).float(), self.cuda_)
        values = self.value(obs, act)
        return values


    def train_process(self, stat, trainer):
        info = {}
        state = trainer.env.reset()
        if self.args.reward_record_type is 'episode_mean_step':
            trainer.mean_reward = 0
            trainer.mean_success = 0

        for t in range(self.args.max_steps):
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            action_out = self.policy(state_, info=info, stat=stat)
            action = select_action(self.args, action_out, status='train', info=info)
            _, actual = translate_action(self.args, action, trainer.env)
            next_state, reward, done, debug = trainer.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            trans = self.Transition(state,
                                    action.cpu().numpy(),
                                    np.array(reward),
                                    next_state,
                                    done,
                                    done_
                                   )
            self.transition_update(trainer, trans, stat)
            success = debug['success'] if 'success' in debug else 0.0
            trainer.steps += 1
            if self.args.reward_record_type is 'mean_step':
                trainer.mean_reward = trainer.mean_reward + 1/trainer.steps*(np.mean(reward) - trainer.mean_reward)
                trainer.mean_success = trainer.mean_success + 1/trainer.steps*(success - trainer.mean_success)
            elif self.args.reward_record_type is 'episode_mean_step':
                trainer.mean_reward = trainer.mean_reward + 1/(t+1)*(np.mean(reward) - trainer.mean_reward)
                trainer.mean_success = trainer.mean_success + 1/(t+1)*(success - trainer.mean_success)
            else:
                raise RuntimeError('Please enter a correct reward record type, e.g. mean_step or episode_mean_step.')
            stat['mean_reward'] = trainer.mean_reward
            stat['mean_success'] = trainer.mean_success
            if done_:
                break
            state = next_state
        stat['turn'] = t + 1
        trainer.episodes += 1


    def unpack_data(self, batch):
        batch_size = len(batch.state)
        rewards = cuda_wrapper(torch.tensor(batch.reward, dtype=torch.float), self.cuda_)
        last_step = cuda_wrapper(torch.tensor(batch.last_step, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        done = cuda_wrapper(torch.tensor(batch.done, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.action))[0], axis=0), dtype=torch.float), self.cuda_)
        state = cuda_wrapper(prep_obs(list(zip(batch.state))), self.cuda_)
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda_)
        return (rewards, last_step, done, actions, state, next_state)
