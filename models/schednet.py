import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from learning_algorithms.reinforce import *
from collections import namedtuple



class SchedNet(Model):

    def __init__(self, args, target_net=None):
        super(SchedNet, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.Transition = namedtuple('Transition', ('state', 'action', 'last_action', 'reward', 'next_state', 'done', 'last_step', 'schedule', 'weight'))

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

    def unpack_data(self, batch):
        batch_size = len(batch.state)
        rewards = cuda_wrapper(torch.tensor(batch.reward, dtype=torch.float), self.cuda_)
        last_step = cuda_wrapper(torch.tensor(batch.last_step, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        done = cuda_wrapper(torch.tensor(batch.done, dtype=torch.float).contiguous().view(-1, 1), self.cuda_)
        actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.action))[0], axis=0), dtype=torch.float), self.cuda_)
        last_actions = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.last_action))[0], axis=0), dtype=torch.float), self.cuda_)
        schedules = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.schedule))[0], axis=0), dtype=torch.float), self.cuda_)
        weights = cuda_wrapper(torch.tensor(np.stack(list(zip(*batch.weight))[0], axis=0), dtype=torch.float), self.cuda_)
        state = cuda_wrapper(prep_obs(list(zip(batch.state))), self.cuda_)
        next_state = cuda_wrapper(prep_obs(list(zip(batch.next_state))), self.cuda_)
        return (rewards, last_step, done, actions, last_actions, state, next_state, schedules, weights)

    def construct_policy_net(self):
        self.action_dict = nn.ModuleDict( {'message_encoder_0': nn.ModuleList([nn.Linear(self.obs_dim, self.hid_dim) for _ in range(self.n_)]),\
                                           'message_encoder_1': nn.ModuleList([nn.Linear(self.hid_dim, self.args.l) for _ in range(self.n_)]),\
                                           'weight_generator_0': nn.ModuleList([nn.Linear(self.obs_dim, self.hid_dim) for _ in range(self.n_)]),\
                                           'weight_generator_1': nn.ModuleList([nn.Linear(self.hid_dim, 1) for _ in range(self.n_)]),\
                                           'action_selector_0': nn.ModuleList([nn.Linear(self.obs_dim+self.args.l*self.n_, self.hid_dim) for _ in range(self.n_)]),\
                                           'action_selector_1': nn.ModuleList([nn.Linear(self.hid_dim, self.act_dim) for _ in range(self.n_)])
                                          }
                                        )

    def construct_value_net(self):
        self.value_dict = nn.ModuleDict( {'share_critic0': nn.Linear(self.obs_dim, self.hid_dim),\
                                          'share_critic1': nn.Linear(self.obs_dim, self.hid_dim),\
                                          'weight_critic': nn.Linear(self.hid_dim+1, 1),\
                                          'action_critic': nn.Linear(self.hid_dim, 1)
                                         }
                                       )

    def construct_model(self):
        self.comm_mask = cuda_wrapper(torch.ones(self.n_, self.n_) - torch.eye(self.n_, self.n_), self.cuda_)
        self.construct_value_net()
        self.construct_policy_net()

    def weight_generator(self, obs):
        batch_size = obs.size(0)
        w = []
        for i in range(self.n_):
            h = torch.relu( self.action_dict['weight_generator_0'][i](obs[:, i, :]) )
            h = self.action_dict['weight_generator_1'][i](h)
            w.append(h)
        w = torch.stack(w, dim=1).contiguous().view(batch_size, self.n_) # shape = (batch_size, n)
        return w

    def weight_based_scheduler(self, w):
        if self.args.schedule is 'top_k':
            _, k_ind = torch.topk(w, self.args.k, dim=-1, sorted=False)
        elif self.args.schedule is 'softmax_k':
            k_ind = torch.multinomial(torch.softmax(w, dim=-1), self.args.k, replacement=False)
            k_ind, _ = torch.sort(k_ind)
        else:
            raise RuntimeError('Please input the the correct schedule, e.g. top_k or softmax_k.')
        onehot_k_ind = cuda_wrapper(torch.zeros_like(w), cuda=self.cuda_)
        onehot_k_ind.scatter_(-1, k_ind, 1)
        return onehot_k_ind

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, gate=None, info={}, stat={}):
        batch_size = obs.size(0)
        m = []
        for i in range(self.n_):
            h = torch.relu( self.action_dict['message_encoder_0'][i](obs[:, i, :]) )
            h = torch.relu( self.action_dict['message_encoder_1'][i](h) )
            m.append(h)
        m = torch.stack(m, dim=1) # shape = (batch_size, n, hid_size)
        m = m.unsqueeze(1).expand(batch_size, self.n_, self.n_, self.args.l) # shape = (batch_size, n, hid_size) -> (batch_size, 1, n, hid_size) -> (batch_size, n, n, hid_size)
        c = schedule # shape = (batch_size, n)
        c = c.unsqueeze(1).expand(batch_size, self.n_, self.n_) # shape = (batch_size, n) -> (batch_size, 1, n) -> (batch_size, n, n)
        c = c.unsqueeze(-1).expand(batch_size, self.n_, self.n_, self.args.l) # shape = (batch_size, n, n) -> (batch_size, n, n, 1) -> (batch_size, n, n, hid_size)
        shared_m = m * c.float()
        shared_m = shared_m.contiguous().view(batch_size, self.n_, self.args.l*self.n_)
        action = []
        for i in range(self.n_):
            h = torch.relu( self.action_dict['action_selector_0'][i]( torch.cat([obs[:, i, :], shared_m[:, i, :]], dim=-1) ) )
            h = self.action_dict['action_selector_1'][i](h)
            action.append(h)
        action = torch.stack(action, dim=1)
        if batch_size == 1:
            stat['schedule'] = schedule.unsqueeze(1).detach().cpu().numpy()
        return action

    def value(self, obs, w, act=None):
        shared_param0 = self.value_dict['share_critic0'](obs)
        shared_param1 = self.value_dict['share_critic1'](obs)
        q = self.value_dict['weight_critic']( torch.cat([shared_param0, w], dim=-1) )
        v = self.value_dict['action_critic'](shared_param1)
        return q.contiguous().view(-1, self.n_), v.contiguous().view(-1, self.n_)

    def get_loss(self, batch):
        batch_size = len(batch.state)
        # collect the transition data
        rewards, last_step, done, actions, last_actions, state, next_state, schedules, weights = self.unpack_data(batch)
        # construct the computational graph
        action_out = self.policy(state, schedule=schedules)
        weight_action_out = self.weight_generator(state)
        q, v = self.value(state, weights.unsqueeze(-1))
        q_, _ = self.value(state, weight_action_out.unsqueeze(-1))
        # get the next actions and the next values
        next_q, next_v = self.target_net.value( next_state, self.target_net.weight_generator(next_state).unsqueeze(-1).detach() )
        returns_q = cuda_wrapper(torch.zeros((batch_size, self.n_), dtype=torch.float), self.cuda_)
        returns_v = cuda_wrapper(torch.zeros((batch_size, self.n_), dtype=torch.float), self.cuda_)
        # calculate the return
        assert returns_v.size() == rewards.size()
        assert returns_q.size() == rewards.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_v[i].detach()
            else:
                next_return = next_v[i].detach()
            returns_v[i] = rewards[i] + self.args.gamma * next_return
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_q[i].detach()
            else:
                next_return = next_q[i].detach()
            returns_q[i] = rewards[i] + self.args.gamma * next_return
        # construct the action loss and the value loss
        deltas_v = returns_v - v
        deltas_q = returns_q - q
        advantages_v = deltas_v.contiguous().view(-1, 1).detach()
        advantages_q = q_.contiguous().view(-1, 1)
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        if self.args.continuous:
            action_means = actions.contiguous().view(-1, self.act_dim)
            action_stds = cuda_wrapper(torch.ones_like(action_means), self.cuda_)
            log_p_a = normal_log_density(actions.detach(), action_means, action_stds)
            log_prob_a = log_p_a
        else:
            log_p_a = action_out
            log_prob_a = multinomials_log_density(actions.detach(), log_p_a).contiguous().view(-1, 1)
        # log_prob_g = weight_action_out.gather(-1, schedules.long()).contiguous().view(-1, 1)
        assert log_prob_a.size() == advantages_v.size()
        action_loss = - advantages_v * log_prob_a - advantages_q
        action_loss = action_loss.sum() / batch_size
        value_loss = ( deltas_v.pow(2).view(-1).sum() + deltas_q.pow(2).view(-1).sum() ) / batch_size
        return action_loss, value_loss, log_p_a

    def train(self, stat, trainer):
        info = {}
        state = trainer.env.reset()
        action = trainer.init_action
        for t in range(self.args.max_steps):
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            weight = self.weight_generator(state_).detach()
            schedule = self.weight_based_scheduler(weight)
            start_step = True if t == 0 else False
            state_ = cuda_wrapper(prep_obs(state).contiguous().view(1, self.n_, self.obs_dim), self.cuda_)
            action_ = action.clone()
            action_out = self.policy(state_, schedule=schedule, last_act=action_, info=info, stat=stat)
            action = select_action(self.args, action_out, status='train', info=info)
            # return the rescaled (clipped) actions
            _, actual = translate_action(self.args, action, trainer.env)
            next_state, reward, done, _ = trainer.env.step(actual)
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            trans = self.Transition(state,
                                    action.cpu().numpy(),
                                    action_.cpu().numpy(),
                                    np.array(reward),
                                    next_state,
                                    done,
                                    done_,
                                    schedule.cpu().numpy(),
                                    weight.cpu().numpy()
                                   )
            if self.args.replay:
                trainer.replay_buffer.add_experience(trans)
                replay_cond = trainer.steps>self.args.replay_warmup\
                 and len(trainer.replay_buffer.buffer)>=self.args.batch_size\
                 and trainer.steps%self.args.behaviour_update_freq==0
                if replay_cond:
                    trainer.replay_process(stat)
            else:
                online_cond = trainer.steps%self.args.behaviour_update_freq==0
                if online_cond:
                    trainer.transition_process(stat, trans)
            if self.args.target:
                target_cond = trainer.steps%self.args.target_update_freq==0
                if target_cond:
                    self.update_target()
            trainer.steps += 1
            trainer.mean_reward = trainer.mean_reward + 1/trainer.steps*(np.mean(reward) - trainer.mean_reward)
            stat['mean_reward'] = trainer.mean_reward
            if done_:
                break
            state = next_state
        trainer.episodes += 1
