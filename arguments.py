from collections import namedtuple
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenario
from gym_wrapper import *
import numpy as np



scenario_name = 'simple_spread'
# scenario_name = 'simple_world_comm'
# scenario_name = 'simple'

# load scenario from script
scenario = scenario.load(scenario_name + ".py").Scenario()
# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=True)

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
                           'action_num',
                           'skip_connection',
                           'decomposition'
                          ]
                 )

args = Args(agent_num=env.get_num_of_agents(),
            hid_size=1024,
            obs_size=np.max(env.get_shape_of_obs()),
            continuous=False,
            action_dim=np.max(env.get_output_shape_of_act()),
            # action_dim=2,
            comm_iters=2,
            init_std=0.1,
            lrate=5e-4,
            batch_size=1024,
            max_steps=50,
            gamma=0.99,
            mean_ratio=0.0,
            normalize_rewards=False,
            advantages_per_action=False,
            value_coeff=1.0,
            entr=0.001,
            action_num=np.max(env.get_input_shape_of_act()),
            skip_connection=False,
            decomposition=False
           )
