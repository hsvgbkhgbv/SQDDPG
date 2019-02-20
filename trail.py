from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenario
import numpy as np
from CommNet import *
from Trainer import *


scenario_name = "simple_world_comm"

# load scenario from script
scenario = scenario.load(scenario_name + ".py").Scenario()
# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=True)

Args = namedtuple('Args', ['agent_num',\
                           'hid_size',\
                           'obs_size',\
                           'continuous',\
                           'action_dim',\
                           'comm_iters',\
                           'action_heads_num',\
                           'init_std',
                           'lrate',
                           'batch_size',
                           'max_steps',
                           'action_num'])
args = Args(agent_num=6,\
            hid_size=10,\
            obs_size=34,\
            continuous=0,\
            action_dim=5,\
            comm_iters=5,\
            action_heads_num=[5],
            init_std=0.2,
            lrate=0.001,
            batch_size=32,
            max_steps=100,
            action_num=[5])
policy_net = CommNet(args)
epoch = 0
train = Trainer(args, policy_net, env)
train.train_batch()
