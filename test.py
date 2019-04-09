import torch
from tester import *
from arguments import *
from ic3net import *
from commnet import *


model = model_map[model_name]

PATH='./exp1/' + scenario_name + '_' + args.training_strategy + '_' + model_name + '.pt'
behaviour_net = model(args)
checkpoint = torch.load(PATH)
behaviour_net.load_state_dict(checkpoint['model_state_dict'])

test = Tester(env(), behaviour_net, args)
episodes = 10
render = True
test.run_game(episodes, render)
