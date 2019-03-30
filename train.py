import numpy as np
from commnet import *
from trainer import *
import torch
from arguments import *
import os
from ic3net import *

if model_name == 'ic3net':
    policy_net = IC3Net(args)
elif model_name == 'commnet':
    policy_net = CommNet(args)
# policy_net = torch.nn.DataParallel(policy_net, device_ids=[0, 1])
epoch = 0

for i in range(args.train_epoch_num):
    train = Trainer(args, policy_net, env(), False)
    train.train_batch()
    print ('This is the epoch: {}, the mean reward is {:2.4f} and the current action loss to be minimized is: {:2.4f}\n'.format(epoch, train.stats['mean_reward'], train.stats['action_loss']))
    epoch += 1
    if i%10 == 9:
        print ('The model is saved!\n')
        torch.save(policy_net, './exp1/' + scenario_name + '_' + args.training_strategy + '_' + model_name + '.pt')
        with open('./exp1/' + scenario_name + '_' + args.training_strategy + '_' + model_name + '.log', 'w+') as file:
            file.write(str(args)+'\n')
            file.write(str(epoch))
