import numpy as np
from commnet import *
from trainer import *
import torch
from arguments import *
import os
from ic3net import *
from util import *
from logger import Logger



logger = Logger('./logs')

model = model_map[model_name]

train = Trainer(args, model, env())

epoch = 0

for i in range(args.train_epoch_num):
    train.train_batch(i)
    print ('This is the epoch: {}, the mean reward is {:2.4f} and the current action loss to be minimized is: {:2.4f}\n'.format(epoch, train.stats['mean_reward'], train.stats['action_loss']))
    epoch += 1

    ### Tensorboard Logging ###
    info = {'mean_reward': train.stats['mean_reward'],'action_loss': train.stats['action_loss'] }
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

    if i%args.target_update_freq == args.target_update_freq-1:
        torch.save({'model_state_dict': train.behaviour_net.state_dict()}, './exp1/' + scenario_name + '_' + args.training_strategy + '_' + model_name + '.pt')
        print ('The model is saved!\n')
        with open('./exp1/' + scenario_name + '_' + args.training_strategy + '_' + model_name + '.log', 'w+') as file:
            file.write(str(args)+'\n')
            file.write(str(epoch))
