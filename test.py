import torch
from tester import *
from arguments import *


model = model_map[model_name]

PATH='./model_save/' + log_name + '/model.pt'
behaviour_net = model(args)
checkpoint = torch.load(PATH, map_location='cpu')
behaviour_net.load_state_dict(checkpoint['model_state_dict'])

test = Tester(env(), behaviour_net, args)
episodes = 10
render = True
test.run_game(episodes, render)
