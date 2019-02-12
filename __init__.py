from torch import nn


class Agent(nn.Moudle):

    def __init__(self):
        super(Agent, self).__init__()
        raise NotImplemented()

    def update(self):
        raise NotImplemented()

    def action(self, obs, info={}):
        raise NotImplemented()
