import torch
from utilities.tester import *
from arguments import *
import argparse



parser = argparse.ArgumentParser(description='Test rl agent.')
parser.add_argument('--save-model-dir', type=str, nargs='?', default='./model_save/', help='Please input the directory of saving model.')
parser.add_argument('--render', type=bool, default=False, help='Please input the flag to control the render.')
argv = parser.parse_args()

model = model_map[model_name]

if argv.save_model_dir[-1] is '/':
    save_path = argv.save_model_dir
else:
    save_path = argv.save_model_dir+'/'

PATH=save_path + log_name + '/model.pt'

if args.target:
    target_net = model(args)
    behaviour_net = model(args, target_net)
else:
    behaviour_net = model(args)

checkpoint = torch.load(PATH, map_location='cpu') if not args.cuda else torch.load(PATH)
behaviour_net.load_state_dict(checkpoint['model_state_dict'])

test = Tester(env(), behaviour_net, args)
episodes = 10
test.run_game(episodes, render=argv.render)
