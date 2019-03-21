from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenario
import numpy as np
from CommNet import *
from Trainer import *
from gym_wrapper import *
import torch
from util import *
import time

scenario_name = 'simple_spread'
# scenario_name = 'simple_world_comm'

# load scenario from script
scenario = scenario.load(scenario_name + ".py").Scenario()
# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=True)

env.mode = 'human'

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
                           'action_num'
                          ]
                 )

args = Args(agent_num=env.get_num_of_agents(),
            hid_size=100,
            obs_size=np.max(env.get_shape_of_obs()),
            continuous=False,
            action_dim=np.max(env.get_output_shape_of_act()),
            comm_iters=10,
            init_std=0.2,
            lrate=0.001,
            batch_size=32,
            max_steps=1000,
            gamma=0.99,
            mean_ratio=0,
            normalize_rewards=True,
            advantages_per_action=False,
            value_coeff=0.001,
            entr=0.001,
            action_num=np.max(env.get_input_shape_of_act())
           )

# policy_net = CommNet(args)

PATH='./exp1/adversary.pt'
policy_net = torch.load(PATH)

# policy_net.load_state_dict(checkpoint['model_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

policy_net.eval()

state = env().reset()

while True:
    # env().render()
    time.sleep(2)
    action_out, value = policy_net.action(state)
    action = select_action(args, action_out)
    _, actual = translate_action(args, env(), action)
    state, reward, done, info = env().step(actual)
    print (actual)
