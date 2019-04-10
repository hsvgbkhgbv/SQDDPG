import numpy as np
from commnet import *
from trainer import *
import torch
from arguments import *
import os
from ic3net import *
from util import *
from logger import Logger



logger = Logger('./logs' + '_' + scenario_name + '_' + args.training_strategy + '_' + model_name)

model = model_map[model_name]

train = Trainer(args, model, env())

for i in range(args.train_epoch_num):
    batch, stat = train.run_batch()
    if i%args.behaviour_update_freq == args.behaviour_update_freq-1:
        stat = train.train_batch(i, batch, stat)
        print ('This is the epoch: {}, the mean reward is {:2.4f} and the current action loss to be minimized is: {:2.4f}\n'.format(i, stat['mean_reward'], stat['action_loss']))
        for tag, value in stat.items():
            if isinstance(value, np.ndarray):
                logger.image_summary(tag, value, i)
            else:
                logger.scalar_summary(tag, value, i)
    if i%args.save_model_freq == args.save_model_freq-1:
        if 'model_save' not in os.listdir('./'):
            os.mkdir('./model_save')
        torch.save({'model_state_dict': train.behaviour_net.state_dict()}, './model_save/' + scenario_name + '_' + args.training_strategy + '_' + model_name + '.pt')
        print ('The model is saved!\n')
        with open('./model_save/' + scenario_name + '_' + args.training_strategy + '_' + model_name + '.log', 'w+') as file:
            file.write(str(args)+'\n')
            file.write(str(i))
