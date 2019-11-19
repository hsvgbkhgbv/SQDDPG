from collections import namedtuple
from utilities.gym_wrapper import *
import numpy as np
from models.coma import *
from aux import *
from environments.traffic_junction_env import TrafficJunctionEnv



'''define the model name'''
model_name = 'coma_fc'

'''define the special property'''
# independentArgs = namedtuple( 'independentArgs', [] )
aux_args = AuxArgs[model_name]()
alias = '_medium'

'''define the scenario name'''
scenario_name = 'traffic_junction'

'''define the environment'''
env = TrafficJunctionEnv()
env = GymWrapper(env)

MergeArgs = namedtuple('MergeArgs', Args._fields+AuxArgs[model_name]._fields)

# under offline trainer if set batch_size=replay_buffer_size=update_freq -> epoch update
args = Args(model_name=model_name,
            agent_num=env.get_num_of_agents(),
            hid_size=128,
            obs_size=np.max(env.get_shape_of_obs()),
            continuous=False,
            action_dim=np.max(env.get_output_shape_of_act()),
            init_std=0.1,
            policy_lrate=1e-4,
            value_lrate=1e-3,
            max_steps=50,
            batch_size=2,
            gamma=0.99,
            normalize_advantages=False,
            entr=1e-4,
            entr_inc=0.0,
            action_num=np.max(env.get_input_shape_of_act()),
            q_func=True,
            train_episodes_num=int(5e3),
            replay=True,
            replay_buffer_size=2,
            replay_warmup=0,
            cuda=True,
            grad_clip=True,
            save_model_freq=100,
            target=True,
            target_lr=1e-1,
            behaviour_update_freq=2,
            critic_update_times=10,
            target_update_freq=2,
            gumbel_softmax=False,
            epsilon_softmax=True,
            online=False,
            reward_record_type='episode_mean_step',
            shared_parameters=False
           )

args = MergeArgs(*(args+aux_args))

log_name = scenario_name + '_' + model_name + alias