from collections import namedtuple
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenario
from utilities.gym_wrapper import *
import numpy as np
from models.commnet import *
from models.ic3net import *
from models.maddpg import *
from models.masddpg import *
from models.coma import *
from models.schednet import *
from aux import *
from environments.traffic_junction_env import TrafficJunctionEnv
from environments.predator_prey_env import PredatorPreyEnv



Model = dict(commnet=CommNet,
             ic3net=IC3Net,
             independent_commnet=IndependentCommNet,
             maddpg=MADDPG,
             masddpg=MASDDPG,
             coma=COMA,
             schednet=SchedNet
            )

AuxArgs = dict(commnet=commnetArgs,
               independent_commnet=commnetArgs,
               ic3net=ic3netArgs,
               maddpg=maddpgArgs,
               masddpg=maddpgArgs,
               coma=comaArgs,
               schednet=schednetArgs
              )

Strategy=dict(commnet='pg',
              independent_commnet='pg',
              ic3net='pg',
              maddpg='pg',
              masddpg='pg',
              coma='pg',
              schednet='pg'
             )

'''define the model name'''
model_name = 'coma'

'''define the special property'''
aux_args = AuxArgs[model_name](0.5,0.02,5,0.8) # coma
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
                           'online'
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
            policy_lrate=5e-4,
            value_lrate=5e-4,
            max_steps=20,
            batch_size=2,
            gamma=0.99,
            normalize_advantages=False,
            entr=1e-2,
            entr_inc=0.0,
            action_num=np.max(env.get_input_shape_of_act()),
            q_func=True,
            train_episodes_num=int(2e5),
            replay=True,
            replay_buffer_size=2,
            replay_warmup=0,
            cuda=True,
            grad_clip=False,
            save_model_freq=100,
            target=True,
            target_lr=1.0,
            behaviour_update_freq=2,
            critic_update_times=5,
            target_update_freq=2,
            gumbel_softmax=False,
            epsilon_softmax=True,
            online=False
           )

args = MergeArgs(*(args+aux_args))

log_name = scenario_name + '_' + model_name + alias
