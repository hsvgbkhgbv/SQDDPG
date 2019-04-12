import torch
from tester import *
from arguments import *
import argparse


parser = argparse.ArgumentParser(description='Test rl agent.')
parser.add_argument('--save-model-dir', type=str, nargs='?', help='Please input the directory of saving model.')
argv = parser.parse_args()

model = model_map[model_name]

PATH=argv.save_model_dir + log_name + '/model.pt'

behaviour_net = model(args)
checkpoint = torch.load(PATH, map_location='cpu')
behaviour_net.load_state_dict(checkpoint['model_state_dict'])

test = Tester(env(), behaviour_net, args)
episodes = 10
render = True
test.run_game(episodes, render)
