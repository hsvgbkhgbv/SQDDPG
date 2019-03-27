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
                           'normalize_rewards',
                           'value_coeff',
                           'entr',
                           'action_num',
                           'skip_connection',
                           'training_strategy'
                          ]
                 )

args = Args(agent_num=env.get_num_of_agents(),
            hid_size=128,
            obs_size=np.max(env.get_shape_of_obs()),
            continuous=False,
            action_dim=np.max(env.get_output_shape_of_act()),
            comm_iters=2,
            init_std=0.1,
            lrate=1e-3,
            batch_size=128,
            max_steps=100,
            gamma=0.99,
            normalize_rewards=True,
            value_coeff=1e-3,
            entr=0.0,
            action_num=np.max(env.get_input_shape_of_act()),
            skip_connection=True,
            training_strategy='actor_critic'
           )
