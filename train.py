import numpy as np
from utilities.trainer import *
import torch
from arguments import *
import os
from utilities.util import *
from utilities.logger import Logger
import argparse



parser = argparse.ArgumentParser(description='Test rl agent.')
parser.add_argument('--save-path', type=str, nargs='?', default='./', help='Please input the directory of saving model.')
# parser.add_argument('--strategy', type=str, nargs='?', default='pg', help='Please input the strategy of learning, such as pg or q.')
# parser.add_argument('--online', action='store_true', help='Please indicate whether the training is online (True) or offline (False).')
argv = parser.parse_args()



if argv.save_path[-1] is '/':
    save_path = argv.save_path
else:
    save_path = argv.save_path+'/'

logger = Logger(save_path+'logs/'+log_name)

model = Model[model_name]

strategy = Strategy[model_name]

print ( '{}\n'.format(args) )

if strategy == 'pg':
    train = PGTrainer(args, model, env(), logger, args.online)
elif strategy == 'q':
    train = QTrainer(args, model, env(), logger)

for i in range(args.train_episodes_num):
    stat = train.run()
    if i%args.save_model_freq == args.save_model_freq-1:
        if 'model_save' not in os.listdir(save_path):
            os.mkdir(save_path+'model_save')
        if log_name not in os.listdir(save_path+'model_save/'):
            os.mkdir(save_path+'model_save/'+log_name)
        torch.save({'model_state_dict': train.behaviour_net.state_dict()}, save_path+'model_save/'+log_name+'/model.pt')
        print ('The model is saved!\n')
        with open(save_path+'model_save/'+log_name +'/log.txt', 'w+') as file:
            file.write(str(args)+'\n')
            file.write(str(i))
