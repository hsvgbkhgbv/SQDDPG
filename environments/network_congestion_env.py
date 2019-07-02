#!/usr/bin/env python
# -*- coding: utf-8 -*-
# origin from https://github.com/IC3Net/IC3Net/blob/master/ic3net-envs/ic3net_envs/predator_prey_env.py
# edit by YuanZhang (2019.03.27)

"""
"""

# core modules
import random
import math

# 3rd party modules
import gym
import numpy as np
from gym import spaces


class NetworkCongestionEnv(gym.Env):

    def __init__(self,):
        self.__version__ = "0.0.1"

        self.n = 20
        self.nroads = 10

        np.random.seed(2019)
        # self.roads = np.random.rand(self.nroads,3)
        # self.roads = np.array([np.random.rand(3)]*self.nroads)
        self.roads = np.array([np.random.rand(1)]*self.nroads)
        np.random.seed()
        print("Road weights",self.roads)

        # gym env
        self.naction = self.nroads # each agent can choose one road
        # self.obs_dim = self.n * self.nroads # state is onehot*n
        self.obs_dim = self.nroads
        self.action_space = []
        self.observation_space = []
        for agent_id in range(self.n):
            # Action for each agent will be naction
            self.action_space.append(spaces.Discrete(self.naction))
            # Observation for each agent will be self.n size
            self.observation_space.append(spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=float))
        # self.action_history = np.array([0.0]*(self.nroads))
        return


    def step(self, actions):
        if self.episode_over:
            raise RuntimeError("Episode is done")

        actions = np.array(actions).squeeze()
        actions = np.atleast_1d(actions)

        self._take_actions(actions)
        self.episode_over = False
        debug = {"cars_on_roads":self.cars_on_roads}
        return self._get_obs(), self._get_reward(), self.episode_over, debug

    def reset(self):
        self.episode_over = False
        self.cars_on_roads = [0]*self.nroads
        # self.action_history = np.array([0.0]*(self.nroads*self.n))
        self.action_history = np.array([0.0]*(self.nroads))
        self.t = 0
        return self._get_obs()

    def _get_obs(self):
        return [self.action_history]*self.n
        # return [np.array(self.cars_on_roads)] * self.n

    def _get_reward(self):
        # costs = [self.roads[i][0] * self.cars_on_roads[i]**2 + self.roads[i][1] * self.cars_on_roads[i] + self.roads[i][2] for i in range(self.nroads)]
        costs = [self.roads[i][0] * self.cars_on_roads[i]**2 for i in range(self.nroads)]
        # reward = -np.sum(np.array(costs) * np.array(self.cars_on_roads))
        reward = -np.sum(np.array(costs))
        # return [reward/self.n]*self.n
        return [reward]*self.n

    def _take_actions(self, actions):
        self.cars_on_roads = [0]*self.nroads
        for a in actions:
            i = np.argmax(a)
            self.cars_on_roads[i] += 1
        # current_action = np.concatenate(actions,axis=None)
        current_action = np.array(self.cars_on_roads)
        self.t += 1
        self.action_history = (1-1./self.t) * self.action_history + 1./self.t * current_action
