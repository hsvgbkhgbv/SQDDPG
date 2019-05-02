from collections import namedtuple



commnetArgs = namedtuple( 'commnetArgs', ['skip_connection', 'comm_iters'] )

# ic3netArgs = namedtuple( 'ic3netArgs', ['comm_iters'] )
ic3netArgs = namedtuple( 'ic3netArgs', [] )

maddpgArgs = namedtuple( 'maddpgArgs', [] )

comaArgs = namedtuple( 'comaArgs', ['softmax_eps_init', 'softmax_eps_end', 'n_step', 'td_lambda'] )

mfqArgs = namedtuple( 'mfqArgs', [] )

mfacArgs = namedtuple( 'mfacArgs', [] )
