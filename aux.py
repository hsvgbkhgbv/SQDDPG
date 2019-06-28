from collections import namedtuple



commnetArgs = namedtuple( 'commnetArgs', ['skip_connection', 'comm_iters'] ) # (bool, int)

ic3netArgs = namedtuple( 'ic3netArgs', [] )

maddpgArgs = namedtuple( 'maddpgArgs', [] )

comaArgs = namedtuple( 'comaArgs', ['softmax_eps_init', 'softmax_eps_end', 'n_step', 'td_lambda'] ) # (bool, float, float, int, float)

mfqArgs = namedtuple( 'mfqArgs', [] )

mfacArgs = namedtuple( 'mfacArgs', [] )

schednetArgs = namedtuple( 'schednetArgs', ['schedule', 'k', 'l'] )

randomArgs = namedtuple( 'randomArgs', [] )

sqpgArgs = namedtuple( 'sqpgArgs', ['sample_size'] )

gcddpgArgs = namedtuple( 'gcddpgArgs', ['sample_size'] )

sqddpgArgs = namedtuple( 'gcddpgArgs', ['sample_size'] )

independentArgs = namedtuple( 'independentArgs', [] )
