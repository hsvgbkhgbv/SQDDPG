import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from collections import namedtuple



class COMA(Model):

    def __init__(self, args, target_net=None):
        super(COMA, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        if self.args.epsilon_softmax:
            self.eps_delta = (args.softmax_eps_init - args.softmax_eps_end) / args.train_episodes_num
            self.eps = args.softmax_eps_init
        self.Transition = namedtuple('Transition', ('state', 'action', 'last_action', 'hidden_state', 'last_hidden_state', 'reward', 'next_state', 'done', 'last_step'))

    def update_eps(self):
        self.eps -= self.eps_delta

    def reload_params_to_target(self):
        self.target_net.action_dict.load_state_dict( self.action_dict.state_dict() )
        self.target_net.value_dict.load_state_dict( self.value_dict.state_dict() )

    def update_target(self):
        params_target_action = list(self.target_net.action_dict.parameters())
        params_behaviour_action = list(self.action_dict.parameters())
        for i in range(len(params_target_action)):
            params_target_action[i] = (1 - self.args.target_lr) * params_target_action[i] + self.args.target_lr * params_behaviour_action[i]
        params_target_value = list(self.target_net.value_dict.parameters())
        params_behaviour_value = list(self.value_dict.parameters())
        for i in range(len(params_target_value)):
            params_target_value[i] = (1 - self.args.target_lr) * params_target_value[i] + self.args.target_lr * params_behaviour_value[i]
        # print ('traget net is updated!\n')

    def unpack_data(self, batch):
        batch_size = len(batch.state)
        rewards = cuda_wrapper(torch.tensor(batch.reward, dtype=torch.float), self.cuda_)
        last_step = cuda_wrapper(torch.tensor(batch.last_step, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        done = cuda_wrapper(torch.tensor(batch.done, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.action))[0], axis=0), dtype=torch.float), self.cuda_)
        last_actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.last_action))[0], axis=0), dtype=torch.float), self.cuda_)
        hidden_states = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.hidden_state))[0], axis=0), dtype=torch.float), self.cuda_)
        last_hidden_states = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.last_hidden_state))[0], axis=0), dtype=torch.float), self.cuda_)
        state = cuda_wrapper(prep_obs(list(zip(batch.state))), self.cuda_)
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda_)
        return (rewards, last_step, done, actions, last_actions, hidden_states, last_hidden_states, state, next_state)

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'transform': nn.ModuleList([nn.Linear(self.obs_dim+self.act_dim, self.hid_dim) for _ in range(self.n_)]),\
                                           'gru_layer': nn.ModuleList([nn.GRUCell(self.hid_dim, self.hid_dim) for _ in range(self.n_)]),\
                                           'action_head': nn.ModuleList([nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_)])
                                          }
                                        )

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict( {'layer_1': nn.ModuleList([nn.Linear( self.obs_dim+self.act_dim*(self.n_-1), self.hid_dim ) for _ in range(self.n_)]),\
                                          'layer_2': nn.ModuleList([nn.Linear(self.hid_dim, self.hid_dim) for _ in range(self.n_)]),\
                                          'value_head': nn.ModuleList([nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_)])
                                         }
                                       )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        batch_size = obs.size(0)
        actions = []
        hs = []
        for i in range(self.n_):
            h = torch.cat( (obs[:, i, :], last_act[:, i, :]), dim=-1 ) # shape=(batch_size, n, act_dim+hid_dim)
            h = h.contiguous().view(batch_size, self.obs_dim+self.act_dim) # shape=(batch_size*n, act_dim+hid_dim)
            h = torch.relu( self.action_dict['transform'][i](h) )
            h = self.action_dict['gru_layer'][i](h, last_hid[:, i, :])
            hs.append(h)
            h = torch.relu(h)
            h = h.contiguous().view(batch_size, self.hid_dim)
            a = self.action_dict['action_head'][i](h)
            actions.append(a)
        a = torch.stack(actions, dim=1)
        return a

    def value(self, obs, act):
        batch_size = obs.size(0)
        act = act.contiguous().view(batch_size, -1)
        values = []
        for i in range(self.n_):
            h = torch.relu( self.value_dict['layer_1'][i]( torch.cat( (obs[:, i, :], act[:, :i*self.act_dim], act[:, (i+1)*self.act_dim:]), dim=-1 ) ) )
            h = torch.relu( self.value_dict['layer_2'][i](h) )
            v = self.value_dict['value_head'][i](h)
            values.append(v)
        values = torch.stack(values, dim=1)
        return values

    def get_loss(self, batch):
        info = {'softmax_eps': self.eps} if self.args.epsilon_softmax else {}
        batch_size = len(batch.state)
        n = self.args.agent_num
        action_dim = self.args.action_dim
        # collect the transition data
        rewards, last_step, done, actions, last_actions, hidden_state, last_hidden_state, state, next_state = self.unpack_data(batch)
        # construct computational graph
        action_out = self.policy(state, last_act=last_actions, last_hid=last_hidden_state, info=info)
        values_ = self.value(state, actions)
        if self.args.q_func:
            values = torch.sum(values_*actions, dim=-1)
        values = values.contiguous().view(-1, n)
        next_action_out = self.target_net.policy(next_state, last_act=actions, last_hid=hidden_state, info=info)
        next_actions = select_action(self.args, next_action_out, status='train', info=info, exploration=False)
        next_values = self.target_net.value(next_state, next_actions)
        # returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        if self.args.q_func:
            next_values = torch.sum(next_values*next_actions, dim=-1)
        next_values = next_values.contiguous().view(-1, n)
        # n-step TD estimate
        assert values.size() == next_values.size()
        returns = td_lambda(rewards, last_step, done, next_values, self.args)
        assert returns.size() == rewards.size()
        # for i in reversed(range(rewards.size(0))):
        #     if last_step[i]:
        #         next_return = 0 if done[i] else next_values[i].detach()
        #     else:
        #         next_return = next_values[i].detach()
        #     returns[i] = rewards[i] + self.args.gamma * next_return
        deltas = returns - values
        # calculate coma
        advantages = ( values - torch.sum(values_*torch.softmax(action_out, dim=-1), dim=-1) ).detach()
        log_p_a = action_out
        log_prob = multinomials_log_density(actions, log_p_a).contiguous().view(-1, 1)
        advantages = advantages.contiguous().view(-1, 1)
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        assert log_prob.size() == advantages.size()
        action_loss = - advantages * log_prob
        action_loss = action_loss.sum() / batch_size
        value_loss = deltas.pow(2).view(-1).sum() / batch_size
        return action_loss, value_loss, log_p_a

    def init_hidden(self, batch_size):
        self.gru_hid = cuda_wrapper(torch.zeros(batch_size, self.n_, self.hid_dim), self.cuda_)

    def get_hidden(self):
        return self.gru_hid.detach()

    def get_episode(self, stat, trainer):
        episode = []
        info = {}
        state = trainer.env.reset()
        action = trainer.init_action
        if self.args.epsilon_softmax:
            info['softmax_eps'] = self.eps
        self.init_hidden(batch_size=1)
        last_hidden_state = self.get_hidden()
        for t in range(self.args.max_steps):
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            action_ = action.clone()
            action_out = self.policy(state_, last_act=action_, last_hid=last_hidden_state, info=info, stat=stat)
            action = select_action(self.args, action_out, status='train', info=info)
            # return the rescaled (clipped) actions
            _, actual = translate_action(self.args, action, trainer.env)
            next_state, reward, done, _ = trainer.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            hidden_state = self.get_hidden()
            trans = self.Transition(state,
                                    action.cpu().numpy(),
                                    action_.cpu().numpy(),
                                    hidden_state.cpu().numpy(),
                                    last_hidden_state.cpu().numpy(),
                                    np.array(reward),
                                    next_state,
                                    done,
                                    done_
                                   )
            last_hidden_state = hidden_state
            episode.append(trans)
            trainer.steps += 1
            trainer.mean_reward = trainer.mean_reward + 1/trainer.steps*(np.mean(reward) - trainer.mean_reward)
            if done_:
                break
            state = next_state
        stat['mean_reward'] = trainer.mean_reward
        trainer.episodes += 1
        if self.args.epsilon_softmax:
            self.update_eps()
        return episode

    def train(self, stat, trainer):
        episode = self.get_episode(stat, trainer)
        if self.args.replay:
            trainer.replay_buffer.add_experience(episode)
            replay_cond = trainer.episodes>self.args.replay_warmup\
             and len(trainer.replay_buffer.buffer)>=self.args.batch_size\
             and trainer.episodes%self.args.behaviour_update_freq==0
            if replay_cond:
                trainer.replay_process(stat)
        else:
            offline_cond = trainer.episodes%self.args.behaviour_update_freq==0
            if offline_cond:
                episode = self.Transition(*zip(*episode))
                trainer.transition_process(stat, episode)
