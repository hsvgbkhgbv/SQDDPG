import torch
from utilities.tester import *
from arguments import *
import argparse


parser = argparse.ArgumentParser(description='Test rl agent.')
parser.add_argument('--save-model-dir', type=str, nargs='?', default='./model_save/', help='Please input the directory of saving model.')
parser.add_argument('--render', action='store_true', help='Please input the flag to control the render.')
parser.add_argument('--episodes', type=int, default=10, help='Please input the number of test episodes')

argv = parser.parse_args()

model = Model[model_name]

strategy = Strategy[model_name]

if argv.save_model_dir[-1] is '/':
    save_path = argv.save_model_dir
else:
    save_path = argv.save_model_dir+'/'

PATH = save_path + log_name + '/model.pt'

if args.target:
    target_net = model(args)
    behaviour_net = model(args, target_net)
else:
    behaviour_net = model(args)

checkpoint = torch.load(PATH, map_location='cpu') if not args.cuda else torch.load(PATH)
behaviour_net.load_state_dict(checkpoint['model_state_dict'])

if strategy == 'pg':
    test = PGTester(env(), behaviour_net, args)
elif strategy == 'q':
    test = QTester(env(), behaviour_net, args)
else:
    raise RuntimeError('Please input the correct strategy, e.g. pg or q.')

print(args)
test.run_game(episodes=argv.episodes, render=argv.render)
test.print_info()
