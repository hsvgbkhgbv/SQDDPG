import numpy as np
import torch
from util import *
import time


class Tester(object):

    def __init__(self, env, behaviour_net, args):
        self.env = env
        self.behaviour_net = behaviour_net.eval()
        self.args = args

    def run_step(self, state):
        state = prep_obs(state).contiguous().view(1, self.args.agent_num, self.args.obs_size)
        action_out = self.behaviour_net.policy(state)
        action = select_action(self.args, action_out, 'test')
        _, actual = translate_action(self.args, action)
        next_state, reward, done, _ = self.env.step(actual)
        return next_state, done

    def run_game(self, episodes, render):
        for ep in range(episodes):
            print ('The episode {} starts!'.format(ep))
            state = self.env.reset()
            while True:
                if render:
                    self.env.render()
                state, done = self.run_step(state)
                time.sleep(0.1)
                if np.all(done):
                    print ('The episode {} is finished!'.format(ep))
                    break
