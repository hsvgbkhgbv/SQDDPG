import numpy as np



def inspector(args):
    if args.model_name is 'maddpg':
        assert args.replay is True
        assert args.q_func is True
        assert args.target is True
        assert args.gumbel_softmax is True
        assert args.online is True
    elif args.model_name is 'commnet':
        assert args.replay is True
        assert hasattr(args, 'comm_iters')
        assert args.comm_iters > 1
        assert args.q_func is False
        assert args.target is False
        assert args.online is False
        assert args.gumbel_softmax is False
        assert args.behaviour_update_freq is args.replay_buffer_size
        assert args.replay_buffer_size is args.batch_size
        assert hasattr(args, 'skip_connection')
    elif args.model_name is 'independent_commnet':
        assert args.replay is True
        assert hasattr(args, 'comm_iters')
        assert args.comm_iters == 1
        assert args.q_func is False
        assert args.target is False
        assert args.online is False
        assert args.gumbel_softmax is False
        assert args.behaviour_update_freq is args.replay_buffer_size
        assert args.replay_buffer_size is args.batch_size
        assert hasattr(args, 'skip_connection')
    elif args.model_name is 'ic3net':
        assert args.replay is True
        assert args.q_func is False
        assert args.target is False
        assert args.online is False
        assert args.gumbel_softmax is False
        assert args.behaviour_update_freq is args.replay_buffer_size
        assert args.replay_buffer_size is args.batch_size
    elif args.model_name is 'coma':
        assert args.replay is True
        assert args.q_func is True
        assert args.target is True
        assert args.online is False
        assert args.continuous is False
        assert args.gumbel_softmax is False
        assert hasattr(args, 'epsilon_softmax')
        assert hasattr(args, 'softmax_eps_init')
        assert hasattr(args, 'softmax_eps_end')
        assert hasattr(args, 'n_step')
        assert hasattr(args, 'td_lambda')
    elif args.model_name is 'mfac':
        assert args.replay is True
        assert args.q_func is True
        assert args.target is True
        assert args.continuous is False
        assert args.online is True
    elif args.model_name is 'schednet':
        assert args.replay is True
        assert args.q_func is True
        assert args.target is True
        assert args.online is True
        assert args.gumbel_softmax is False
        assert hasattr(args, 'schedule')
        assert hasattr(args, 'k')
