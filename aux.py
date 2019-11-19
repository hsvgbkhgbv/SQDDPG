from collections import namedtuple
from models.commnet import *
from models.ic3net import *
from models.maddpg import *
from models.coma import *
from models.schednet import *
from models.sqddpg import *
from models.independent_ac import *
from models.independent_ddpg import *
from models.mfac import *
from models.coma_fc import *



commnetArgs = namedtuple( 'commnetArgs', ['skip_connection', 'comm_iters'] ) # (bool, int)

ic3netArgs = namedtuple( 'ic3netArgs', [] )

maddpgArgs = namedtuple( 'maddpgArgs', [] )

comaArgs = namedtuple( 'comaArgs', ['softmax_eps_init', 'softmax_eps_end', 'n_step', 'td_lambda'] ) # (bool, float, float, int, float)

schednetArgs = namedtuple( 'schednetArgs', ['schedule', 'k', 'l'] )

randomArgs = namedtuple( 'randomArgs', [] )

sqddpgArgs = namedtuple( 'sqddpgArgs', ['sample_size'] )

independentArgs = namedtuple( 'independentArgs', [] )

mfacArgs = namedtuple( 'mfacArgs', [] )

comafcArgs = namedtuple( 'comafcArgs', [] )



Model = dict(commnet=CommNet,
             ic3net=IC3Net,
             independent_commnet=IndependentCommNet,
             maddpg=MADDPG,
             coma=COMA,
             schednet=SchedNet,
             sqddpg=SQDDPG,
             independent_ac=IndependentAC,
             independent_ddpg=IndependentDDPG,
             mfac=MFAC,
             coma_fc=COMAFC
            )



AuxArgs = dict(commnet=commnetArgs,
               independent_commnet=commnetArgs,
               ic3net=ic3netArgs,
               maddpg=maddpgArgs,
               coma=comaArgs,
               schednet=schednetArgs,
               sqddpg=sqddpgArgs,
               independent_ac=independentArgs,
               independent_ddpg=independentArgs,
               mfac=mfacArgs,
               coma_fc=comafcArgs
              )



Strategy=dict(commnet='pg',
              independent_commnet='pg',
              ic3net='pg',
              maddpg='pg',
              coma='pg',
              schednet='pg',
              sqddpg='pg',
              independent_ac='pg',
              independent_ddpg='pg',
              mfac='pg',
              coma_fc='pg'
             )



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
                           'reward_record_type',
                           'shared_parameters' # boolean
                          ]
                 )
