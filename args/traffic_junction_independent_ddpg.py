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
from models.independent_ddpg import *
from aux import *
from environments.traffic_junction_env import TrafficJunctionEnv
from environments.predator_prey_env import PredatorPreyEnv
from environments.network_congestion_env import NetworkCongestionEnv



Model = dict(commnet=CommNet,
             ic3net=IC3Net,
             maddpg=MADDPG,
             coma=COMA,
             schednet=SchedNet,
             independent_ddpg=IndependentDDPG
            )

AuxArgs = dict(commnet=commnetArgs,
               ic3net=ic3netArgs,
               maddpg=maddpgArgs,
               coma=comaArgs,
               schednet=schednetArgs,
               independent_ddpg=maddpgArgs
              )

Strategy=dict(commnet='pg',
              ic3net='pg',
              maddpg='pg',
              coma='pg',
              schednet='pg',
              independent_ddpg='pg'
             )

'''define the model name'''
model_name = 'independent_ddpg'

'''define the special property'''
aux_args = AuxArgs[model_name]() 
alias = ''

'''define the scenario name'''
scenario_name = 'traffic_junction' 

'''define the environment'''
env = TrafficJunctionEnv()
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
                           'reward_record_type'
                          ]
                 )

MergeArgs = namedtuple('MergeArgs', Args._fields+AuxArgs[model_name]._fields)

# under offline trainer if set batch_size=replay_buffer_size=update_freq -> epoch update
args = Args(model_name=model_name,
            agent_num=env.get_num_of_agents(),
            hid_size=128,
            obs_size=np.max(env.get_shape_of_obs()),
            continuous=False,
            action_dim=np.max(env.get_output_shape_of_act()),
            init_std=0.1,
            policy_lrate=1e-3,
            value_lrate=1e-3,
            max_steps=50,
            batch_size=100,
            gamma=0.99,
            normalize_advantages=False,
            entr=1e-2,
            entr_inc=0.0,
            action_num=np.max(env.get_input_shape_of_act()),
            q_func=True,
            train_episodes_num=int(1e4),
            replay=True,
            replay_buffer_size=1e4,
            replay_warmup=0,
            cuda=True,
            grad_clip=False,
            save_model_freq=100,
            target=True,
            target_lr=1.0,
            behaviour_update_freq=100,
            critic_update_times=1,
            target_update_freq=1000,
            gumbel_softmax=True,
            epsilon_softmax=False,
            online=True,
            reward_record_type='episode_mean_step'
           )

args = MergeArgs(*(args+aux_args))

log_name = scenario_name + '_' + model_name + alias
