import torch
from tester import *
from arguments import *
from ic3net import *
from commnet import *


model_map = dict(commnet=CommNet,
                 ic3net=IC3Net
)

model = model_map[model_name]

PATH='./exp1/' + scenario_name + '_' + args.training_strategy + '_' + model_name + '.pt'
behaviour_net = torch.load(PATH, map_location='cpu')
behaviour_net.cuda_ = False

test = Tester(env(), behaviour_net, args)
episodes = 10
render = True
test.run_game(episodes, render)
