import numpy as np
import torch
from util import *
import time


class Tester(object):

    def __init__(self, env, policy_net, args):
        self.env = env
        self.policy_net = policy_net.eval()
        self.args = args

    def run_step(self, state):
        action_out, value = self.policy_net.action(state)
        action = select_action(self.args, action_out, 'test')
        _, actual = translate_action(self.args, self.env, action)
        next_state, reward, done, info = self.env.step(actual)
        return next_state, done

    def run_game(self, episodes, render):
        for ep in range(episodes):
            print ('The episode {} starts!'.format(ep))
            state = self.env.reset()
            while True:
                if render:
                    self.env.render()
                time.sleep(1)
                state, done = self.run_step(state)
                if np.all(done):
                    print ('The episode {} is finished!'.format(ep))
                    break
