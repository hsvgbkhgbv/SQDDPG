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
                           'action_num',
                           'gamma',
                           'mean_ratio',
                           'normalize_rewards',
                           'advantages_per_action',
                           'value_coeff',
                           'entr'])
args = Args(agent_num=6,
            hid_size=10,
            obs_size=34,
            continuous=0,
            action_dim=1,
            comm_iters=5,
            init_std=0.2,
            lrate=0.001,
            batch_size=32,
            max_steps=100,
            action_num=5,
            gamma=0.99,
            mean_ratio=0,
            normalize_rewards=1,
            advantages_per_action=0,
            value_coeff=0.001,
            entr=0.001)
policy_net = CommNet(args)
epoch = 0
while True:
    train = Trainer(args, policy_net, env)
    train.train_batch()
    print ('This is the epoch: {} and the current advantage is: {}'.format(epoch, train.stats['action_loss']))
    epoch += 1