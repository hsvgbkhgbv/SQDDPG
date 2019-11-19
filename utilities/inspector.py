import numpy as np



def inspector(args):
    if args.model_name is 'maddpg':
        assert args.replay is True
        assert args.q_func is True
        assert args.target is True
        assert args.gumbel_softmax is True
        assert args.epsilon_softmax is False
        assert args.online is True
    elif args.model_name is 'independent_ac':
        assert args.replay is True
        assert args.q_func is True
        assert args.target is True
        assert args.online is True
        assert args.gumbel_softmax is False
        assert args.epsilon_softmax is False
    elif args.model_name is 'independent_ddpg':
        assert args.replay is True
        assert args.q_func is False
        assert args.target is True
        assert args.online is True
        assert args.gumbel_softmax is True
        assert args.epsilon_softmax is False
    elif args.model_name is 'sqddpg':
        assert args.replay is True
        assert args.q_func is True
        assert args.target is True
        assert args.gumbel_softmax is True
        assert args.epsilon_softmax is False
        assert args.online is True
        assert hasattr(args, 'sample_size')
    elif args.model_name is 'coma_fc':
        assert args.replay is True
        assert args.q_func is True
        assert args.target is True
        assert args.online is True
        assert args.continuous is False
        assert args.gumbel_softmax is False
        assert args.epsilon_softmax is False
    else:
        raise NotImplementedError('The model is not added!')
