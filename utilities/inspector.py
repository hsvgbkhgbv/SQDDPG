import numpy as np



def inspector(model_name, args):
    if model_name == 'maddpg':
        assert args.replay == True
        assert args.q_func == True
        assert args.target == True
    elif model_name == 'commnet':
        assert args.replay == False
        assert args.comm_iters > 1
        assert args.q_func == False
    elif model_name == 'independent_commnet':
        assert args.replay == False
        assert args.comm_iters == 1
        assert args.q_func == False
    elif model_name == 'ic3net':
        assert args.replay == False
        assert args.comm_iters > 1
        assert args.q_func == False
    elif model_name == 'independent_ic3net':
        assert args.replay == False
        assert args.comm_iters == 1
        assert args.q_func == False
