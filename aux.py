from collections import namedtuple



commnetArgs = namedtuple( 'commnetArgs', ['skip_connection', 'comm_iters'] )

ic3netArgs = namedtuple( 'ic3netArgs', ['comm_iters'] )

maddpgArgs = namedtuple( 'maddpgArgs', ['gumbel_softmax'] )

comaArgs = namedtuple( 'comaArgs', ['epsilon_softmax', 'softmax_eps_init', 'softmax_eps_end'] )
