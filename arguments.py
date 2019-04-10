from collections import namedtuple
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenario
from gym_wrapper import *
import numpy as np
from commnet import *
from ic3net import *
from maddpg import *



model_map = dict(commnet=CommNet,
                 ic3net=IC3Net,
                 independent_commnet=IndependentCommNet,
                 independent_ic3net=IndependentIC3Net,
                 maddpg=MADDPG
)

# model_name = 'commnet'
# model_name = 'ic3net'
# model_name = 'independent_commnet'
# model_name = 'independent_ic3net'
model_name = 'maddpg'

scenario_name = 'simple_spread'
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
                           'epoch_size',
                           'max_steps',
                           'gamma',
                           'normalize_advantages',
                           'value_coeff',
                           'entr',
                           'action_num',
                           'skip_connection',
                           'training_strategy',
                           'train_epoch_num',
                           'replay_buffer_size',
                           'replay_iters',
                           'cuda',
                           'grad_clip',
                           'target_lr',
                           'target_update_freq',
                           'save_model_freq'
                          ]
                 )

args = Args(agent_num=env.get_num_of_agents(),
            hid_size=64,
            obs_size=np.max(env.get_shape_of_obs()),
            continuous=False,
            action_dim=np.max(env.get_output_shape_of_act()),
            comm_iters=2,
            init_std=0.1,
            lrate=1e-2,
            epoch_size=32,
            max_steps=50,
            gamma=0.99,
            normalize_advantages=False,
            value_coeff=10.0,
            entr=1e-3,
            action_num=np.max(env.get_input_shape_of_act()),
            skip_connection=True,
            training_strategy='ddpg',
            train_epoch_num=10000,
            replay_buffer_size=1e6,
            replay_iters=1,
            cuda=False,
            grad_clip=True,
            target_lr=1e-1,
            target_update_freq=10,
            save_model_freq=10
           )
