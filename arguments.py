from collections import namedtuple
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenario
from utilities.gym_wrapper import *
import numpy as np
from models.commnet import *
from models.ic3net import *
from models.maddpg import *
from models.coma import *
from models.mf import *
from aux import *
from environments.traffic_junction_env import TrafficJunctionEnv
from environments.predator_prey_env import PredatorPreyEnv



model_map = dict(commnet=CommNet,
                 ic3net=IC3Net,
                 independent_commnet=IndependentCommNet,
                 independent_ic3net=IndependentIC3Net,
                 maddpg=MADDPG,
                 coma=COMA,
                 mfac=MFAC,
                 mfq=MFQ
                )

AuxArgs = dict(commnet=commnetArgs,
               independent_commnet=commnetArgs,
               ic3net=ic3netArgs,
               independent_ic3net=ic3netArgs,
               maddpg=maddpgArgs,
               coma=comaArgs,
               mfac=mfacArgs,
               mfq=mfqArgs
              )

'''define the model name'''
# model_name = 'commnet'
# model_name = 'ic3net'
# model_name = 'independent_commnet'
# model_name = 'independent_ic3net'
model_name = 'maddpg'
# model_name = 'coma'
# model_name = 'mfac'
# model_name = 'mfq'

'''define the scenario name'''
scenario_name = 'simple_spread'
# scenario_name = 'simple'

'''define the special property'''
# commnetArgs = namedtuple( 'commnetArgs', ['skip_connection', 'comm_iters'] )
# ic3netArgs = namedtuple( 'ic3netArgs', ['comm_iters'] )
# maddpgArgs = namedtuple( 'maddpgArgs', [] )
# comaArgs = namedtuple( 'comaArgs', ['softmax_eps_init', 'softmax_eps_end', 'n_step', 'td_lambda'] )
# mfacArgs = namedtuple( 'mfacArgs', [] )
mfqArgs = namedtuple( 'mfqArgs', [] )

aux_args = AuxArgs[model_name]()
alias = ''

'''load scenario from script'''
scenario = scenario.load(scenario_name+".py").Scenario()

'''create world'''
world = scenario.make_world()

'''create multiagent environment'''
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=True)
env = GymWrapper(env)

Args = namedtuple('Args', ['model_name',
                           'agent_num',
                           'hid_size',
                           'obs_size',
                           'continuous',
                           'action_dim',
                           'init_std',
                           'policy_lrate',
                           'value_lrate',
                           'epoch_size',
                           'max_steps',
                           'batch_size',
                           'gamma',
                           'normalize_advantages',
                           'entr',
                           'action_num',
                           'q_func',
                           'train_epoch_num',
                           'replay',
                           'replay_buffer_size',
                           'replay_iters',
                           'replay_warmup',
                           'cuda',
                           'grad_clip',
                           'save_model_freq',
                           'target',
                           'target_lr',
                           'target_update_freq',
                           'epsilon_softmax',
                           'gumbel_softmax'
                          ]
                 )

MergeArgs = namedtuple( 'MergeArgs', Args._fields+AuxArgs[model_name]._fields )

args = Args(model_name=model_name,
            agent_num=env.get_num_of_agents(),
            hid_size=64,
            obs_size=np.max(env.get_shape_of_obs()),
            continuous=False,
            action_dim=np.max(env.get_output_shape_of_act()),
            init_std=0.1,
            policy_lrate=1e-3,
            value_lrate=1e-2,
            epoch_size=1,
            max_steps=100,
            batch_size=128,
            gamma=0.95,
            normalize_advantages=False,
            entr=1e-3,
            action_num=np.max(env.get_input_shape_of_act()),
            q_func=True,
            train_epoch_num=10000,
            replay=True,
            replay_buffer_size=1e5,
            replay_iters=1,
            replay_warmup=20,
            cuda=True,
            grad_clip=True,
            save_model_freq=10,
            target=True,
            target_lr=1e-2,
            target_update_freq=1,
            epsilon_softmax=False,
            gumbel_softmax=True
           )

args = MergeArgs(*(args+aux_args))

log_name = scenario_name + '_' + model_name + alias
