import numpy as np
import torch
from utilities.util import *
import time
import signal
import sys

class PGTester(object):

    def __init__(self, env, behaviour_net, args):
        self.env = env
        self.behaviour_net = behaviour_net.cuda().eval() if args.cuda else behaviour_net.eval()
        self.args = args
        self.cuda_ = self.args.cuda and torch.cuda.is_available()

    def action_logits(self, state, schedule, last_action, last_hidden, info):
        return self.behaviour_net.policy(state, schedule=schedule, last_act=last_action, last_hid=last_hidden, info=info)

    def run_step(self, state, schedule, last_action, last_hidden, info={}):
        state = cuda_wrapper(prep_obs(state).contiguous().view(1, self.args.agent_num, self.args.obs_size), cuda=self.cuda_)
        if self.args.model_name in ['schednet']:
            weight = self.behaviour_net.weight_generator(state).detach()
            schedule, _ = self.behaviour_net.weight_based_scheduler(weight, exploration=False)
        action_out = self.action_logits(state, schedule, last_action, last_hidden, info)
        action = select_action(self.args, action_out, status='test')
        _, actual = translate_action(self.args, action, self.env)
        next_state, reward, done, debug  = self.env.step(actual)

        success = debug['success'] if 'success' in debug else 0.0
        disp = 'The rewards of agents are:'
        for r in reward:
            disp += ' '+str(r)[:7]
        print (disp+'.')
        return next_state, action, done, reward, success

    def run_game(self, episodes, render):
        action = cuda_wrapper(torch.zeros((1, self.args.agent_num, self.args.action_dim)), cuda=self.cuda_)
        info = {}
        if render and self.env.name in ['traffic_junction','predator_prey']:
            signal.signal(signal.SIGINT, self.signal_handler)
            self.env.init_curses()

        if self.args.model_name in ['coma', 'ic3net']:
            self.behaviour_net.init_hidden(batch_size=1)
            last_hidden = self.behaviour_net.get_hidden()
        else:
            last_hidden = None
        if self.args.model_name in ['ic3net']:
            gate = self.behaviour_net.gate(last_hidden[:, :, :self.args.hid_size])
            schedule = self.behaviour_net.schedule(gate)
        else:
            schedule = None

        self.all_reward = []
        self.all_turn = []
        self.all_success = [] # special for traffic junction
        for ep in range(episodes):
            print ('The episode {} starts!'.format(ep))
            episode_reward = []
            episode_success = []
            state = self.env.reset()
            t = 0
            while True:
                if render:
                    self.env.render()
                state, action, done, reward, success = self.run_step(state, schedule, action, last_hidden, info=info)
                if self.args.model_name in ['coma']:
                    last_hidden = self.behaviour_net.get_hidden()
                episode_reward.append(np.mean(reward))
                episode_success.append(success)
                if render:
                    time.sleep(0.01)
                if np.all(done) or t==self.args.max_steps-1:

                    print ('The episode {} is finished!'.format(ep))
                    self.all_reward.append(np.mean(episode_reward))
                    self.all_success.append(np.mean(episode_success))
                    self.all_turn.append(t+1)
                    break
                t += 1

    def signal_handler(self, signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        self.env.exit_render()
        sys.exit(0)

    def print_info(self):
        episodes = len(self.all_reward)
        print("\n"+"="*10+ " REUSLTS "+ "="*10)
        print ('Episode: {:4d}'.format(episodes))
        print('Mean Reward: {:2.4f}/{:2.4f}'.format(np.mean(self.all_reward),np.std(self.all_reward)))
        print('Mean Success: {:2.4f}/{:2.4f}'.format(np.mean(self.all_success),np.std(self.all_success)))
        print('Mean Turn: {:2.4f}/{:2.4f}'.format(np.mean(self.all_turn),np.std(self.all_turn)))


class QTester(PGTester):

    def __init__(self, env, behaviour_net, args):
        super(QTester, self).__init__(env, behaviour_net, args)

    def action_logits(self, state, last_action, last_hidden, info):
        return self.behaviour_net.value(state, last_action)
