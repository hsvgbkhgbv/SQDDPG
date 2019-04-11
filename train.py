import numpy as np
from trainer import *
import torch
from arguments import *
import os
from utilities.util import *
from utilities.logger import Logger



logger = Logger('./logs/' + log_name)

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
        if log_name not in os.listdir('./model_save/'):
            os.mkdir('./model_save/'+log_name)
        torch.save({'model_state_dict': train.behaviour_net.state_dict()}, './model_save/' + log_name + '/model.pt')
        print ('The model is saved!\n')
        with open('./model_save/' + log_name + '/log.txt', 'w+') as file:
            file.write(str(args)+'\n')
            file.write(str(i))
