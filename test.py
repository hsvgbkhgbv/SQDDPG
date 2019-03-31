import torch
from tester import *
from arguments import *


PATH='./exp1/' + scenario_name + '_' + args.training_strategy + '_' + model_name + '.pt'
behaviour_net = torch.load(PATH)

test = Tester(env(), behaviour_net, args)
episodes = 10
render = True
test.run_game(episodes, render)
