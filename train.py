import numpy as np
from utilities.trainer import *
import torch
from arguments import *
import os
from utilities.util import *
from utilities.logger import Logger
import argparse
from utilities.multi_processing import MultiProcessTrainer

parser = argparse.ArgumentParser(description='Test rl agent.')
parser.add_argument('--save-path', type=str, nargs='?', default='./', help='Please input the directory of saving model.')
parser.add_argument('--nprocesses', type=int, default=1, help='number of multiprocessing training')
argv = parser.parse_args()

# Force to use cpu for multiprocessing
if argv.nprocesses > 1:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

if argv.save_path[-1] is '/':
    save_path = argv.save_path
else:
    save_path = argv.save_path+'/'

logger = Logger(save_path+'logs/'+log_name)

model = model_map[model_name]

print ( '{}\n'.format(args) )


if argv.nprocesses > 1: 
    train = MultiProcessTrainer(args, argv, lambda: Trainer(args, model, env))
else:
    train = Trainer(args, model, env)

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
        if 'model_save' not in os.listdir(save_path):
            os.mkdir(save_path+'model_save')
        if log_name not in os.listdir(save_path+'model_save/'):
            os.mkdir(save_path+'model_save/'+log_name)
        
        PATH = save_path+'model_save/'+log_name+'/model.pt'
        train.save_model(PATH)
        print ('The model is saved!\n')
        with open(save_path+'model_save/'+log_name +'/log.txt', 'w+') as file:
            file.write(str(args)+'\n')
            file.write(str(i))
