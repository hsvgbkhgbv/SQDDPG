from collections import namedtuple
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenario
from utilities.gym_wrapper import *
import numpy as np
from models.commnet import *
from models.ic3net import *
from models.maddpg import *
from models.coma import *
from models.schednet import *
from models.sqpg import *
from models.gcddpg import *
from models.sqddpg import *
from models.independent import *
from models.independent_ddpg import *
from aux import *
from environments.traffic_junction_env import TrafficJunctionEnv
from environments.predator_prey_env import PredatorPreyEnv



Model = dict(commnet=CommNet,
             ic3net=IC3Net,
             independent_commnet=IndependentCommNet,
             maddpg=MADDPG,
             sqpg=SQPG,
             coma=COMA,
             schednet=SchedNet,
             gcddpg=GCDDPG,
             sqddpg=SQDDPG,
             independent=Independent,
             independent_ddpg=IndependentDDPG
            )

AuxArgs = dict(commnet=commnetArgs,
               independent_commnet=commnetArgs,
               ic3net=ic3netArgs,
               maddpg=maddpgArgs,
               sqpg=sqpgArgs,
               coma=comaArgs,
               schednet=schednetArgs,
               gcddpg=gcddpgArgs,
               sqddpg=sqddpgArgs,
               independent=independentArgs,
               independent_ddpg=independentArgs
              )

Strategy=dict(commnet='pg',
              independent_commnet='pg',
              ic3net='pg',
              maddpg='pg',
              sqpg='pg',
              coma='pg',
              schednet='pg',
              gcddpg='pg',
              sqddpg='pg',
              independent='pg',
              independent_ddpg='pg'
             )

'''define the model name'''
model_name = 'sqddpg'


'''define the scenario name'''
scenario_name = 'simple_spread'

'''define the special property'''
# sqddpgArgs = namedtuple('sqddpgArgs', ['sample_size'])
aux_args = AuxArgs[model_name](1)
alias = '_6_agents_5'

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
                           'max_steps',
                           'batch_size', # steps<-online/episodes<-offline
                           'gamma',
                           'normalize_advantages',
                           'entr',
                           'entr_inc',
                           'action_num',
                           'q_func',
                           'train_episodes_num',
                           'replay',
                           'replay_buffer_size',
                           'replay_warmup',
                           'cuda',
                           'grad_clip',
                           'save_model_freq', # episodes
                           'target',
                           'target_lr',
                           'behaviour_update_freq', # steps<-online/episodes<-offline
                           'critic_update_times',
                           'target_update_freq', # steps<-online/episodes<-offline
                           'gumbel_softmax',
                           'epsilon_softmax',
                           'online',
                           'reward_record_type', # 'mean_step', 'episode_mean_step'
                           'shared_parameters' # boolean
                          ]
                 )

MergeArgs = namedtuple('MergeArgs', Args._fields+AuxArgs[model_name]._fields)

# under offline trainer if set batch_size=replay_buffer_size=update_freq -> epoch update
args = Args(model_name=model_name,
            agent_num=env.get_num_of_agents(),
            hid_size=64,
            obs_size=np.max(env.get_shape_of_obs()),
            continuous=False,
            action_dim=np.max(env.get_output_shape_of_act()),
            init_std=0.1,
            policy_lrate=1e-4,
            value_lrate=1e-3,
            max_steps=200,
            batch_size=32,
            gamma=0.9,
            normalize_advantages=False,
            entr=1e-3,
            entr_inc=0.0,
            action_num=np.max(env.get_input_shape_of_act()),
            q_func=True,
            train_episodes_num=int(2e4),
            replay=True,
            replay_buffer_size=1e6,
            replay_warmup=0,
            cuda=True,
            grad_clip=True,
            save_model_freq=10,
            target=True,
            target_lr=1e-1,
            behaviour_update_freq=100,
            critic_update_times=10,
            target_update_freq=200,
            gumbel_softmax=True,
            epsilon_softmax=False,
            online=True,
            reward_record_type='episode_mean_step',
            shared_parameters=False
           )

args = MergeArgs(*(args+aux_args))

log_name = scenario_name + '_' + model_name + alias
