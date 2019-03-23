import numpy as np
from commnet import *
from trainer import *
import torch
from arguments import *
import os


policy_net = CommNet(args)
num_epoch = 500
epoch = 0

with open(scenario_name+'.log', 'w+') as file:
    file.write(str(args)+'\n')
    file.write(str(num_epoch))

for i in range(num_epoch):
    train = Trainer(args, policy_net, env(), False)
    train.train_batch()
    print ('This is the epoch: {}, the mean reward is {} and the current advantage is: {}\n'\
    .format(epoch, train.stats['mean_reward'], train.stats['action_loss']))
    epoch += 1
    if i%10 == 9:
        print ('The model is saved!\n')
        torch.save(policy_net, './exp1/'+scenario_name+'.pt')
