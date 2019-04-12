from collections import namedtuple
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenario
from utilities.gym_wrapper import *
import numpy as np
from models.commnet import *
from models.ic3net import *
from models.maddpg import *
from aux import *



model_map = dict(commnet=CommNet,
                 ic3net=IC3Net,
                 independent_commnet=IndependentCommNet,
                 independent_ic3net=IndependentIC3Net,
                 maddpg=MADDPG
)

AuxArgs = dict(commnet=commnetArgs,
               independent_commnet=commnetArgs,
               ic3net=ic3netArgs,
               independent_ic3net=ic3netArgs,
               maddpg=maddpgArgs
              )

'''define the model name'''
# model_name = 'commnet'
# model_name = 'ic3net'
model_name = 'independent_commnet'
# model_name = 'independent_ic3net'
# model_name = 'maddpg'

'''define the scenario name'''
# scenario_name = 'simple_spread'
scenario_name = 'simple'

'''define the training strategy'''
training_strategy='actor_critic'

'''define the special property'''
# commnetArgs = namedtuple( 'commnetArgs', ['skip_connection', 'comm_iters'] )
# ic3netArgs = namedtuple( 'ic3netArgs', ['comm_iters'] )
# maddpgArgs = namedtuple( 'maddpgArgs', [] )
aux_args = AuxArgs[model_name](skip_connection=False, comm_iters=2)
alias = '_q_func'

'''load scenario from script'''
scenario = scenario.load(scenario_name + ".py").Scenario()

'''create world'''
world = scenario.make_world()

'''create multiagent environment''' 
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=True)
env = GymWrapper(env)

Args = namedtuple('Args', ['agent_num',
                           'hid_size',
                           'obs_size',
                           'continuous',
                           'action_dim',
                           'init_std',
                           'policy_lrate',
                           'value_lrate',
                           'epoch_size',
                           'max_steps',
                           'gamma',
                           'normalize_advantages',
                           'entr',
                           'action_num',
                           'training_strategy',
                           'q_func',
                           'train_epoch_num',
                           'replay_buffer_size',
                           'replay_iters',
                           'cuda',
                           'grad_clip',
                           'behaviour_update_freq',
                           'save_model_freq',
                           'replay', 
                           'target_lr',
                           'target_update_freq'
                          ]
                 )

MergeArgs = namedtuple( 'MergeArgs', Args._fields+AuxArgs[model_name]._fields )

args = Args(agent_num=env.get_num_of_agents(),
            hid_size=64,
            obs_size=np.max(env.get_shape_of_obs()),
            continuous=False,
            action_dim=np.max(env.get_output_shape_of_act()),
            init_std=0.1,
            policy_lrate=1e-2,
            value_lrate=2e-2,
            epoch_size=32,
            max_steps=50,
            gamma=0.95,
            normalize_advantages=False,
            entr=1e-3,
            action_num=np.max(env.get_input_shape_of_act()),
            training_strategy=training_strategy,
            q_func=False,
            train_epoch_num=10000,
            replay_buffer_size=1e6,
            replay_iters=10,
            cuda=True,
            grad_clip=True,
            behaviour_update_freq=1,
            save_model_freq=10,
            replay=True,
            target_lr=1e-2,
            target_update_freq=2
           )

args = MergeArgs(*(args+aux_args))

log_name = scenario_name + '_' + args.training_strategy + '_' + model_name + alias
