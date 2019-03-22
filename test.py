from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenario
import torch
from util import *
import time
from tester import *
from gym_wrapper import *
from collections import namedtuple


scenario_name = 'simple_spread'
# scenario_name = 'simple_world_comm'

# load scenario from script
scenario = scenario.load(scenario_name + ".py").Scenario()
# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=True)

env.mode = 'human'

env = GymWrapper(env)

Args = namedtuple('Args', ['agent_num',
                           'hid_size',
                           'obs_size',
                           'continuous',
                           'action_dim',
                           'comm_iters',
                           'init_std',
                           'lrate',
                           'batch_size',
                           'max_steps',
                           'gamma',
                           'mean_ratio',
                           'normalize_rewards',
                           'advantages_per_action',
                           'value_coeff',
                           'entr',
                           'action_num'
                          ]
                 )

args = Args(agent_num=env.get_num_of_agents(),
            hid_size=32,
            obs_size=np.max(env.get_shape_of_obs()),
            continuous=False,
            action_dim=np.max(env.get_output_shape_of_act()),
            comm_iters=10,
            init_std=0.01,
            lrate=0.00001,
            batch_size=32,
            max_steps=1000,
            gamma=0.99,
            mean_ratio=0.0,
            normalize_rewards=False,
            advantages_per_action=False,
            value_coeff=0.0,
            entr=0.0,
            action_num=np.max(env.get_input_shape_of_act())
           )

PATH='./exp1/coop1.pt'
policy_net = torch.load(PATH)

test = Tester(env(), policy_net, args)
episodes = 10
render = True
test.run_game(episodes, render)
