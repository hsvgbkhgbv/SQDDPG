import numpy as np


class TransReplayBuffer(object):

    def __init__(self, size):
        self.size = size
        self.buffer = []

    def get_single(self, index):
        return self.buffer[index]

    def offset(self):
        self.buffer.pop(0)
        # self.buffer = []

    def get_batch(self, batch_size):
        length = len(self.buffer)
        indices = np.random.choice(length, batch_size, replace=False)
        batch_buffer = [self.buffer[i] for i in indices]
        return batch_buffer

    def add_experience(self, trans):
        est_len = 1 + len(self.buffer)
        if est_len > self.size:
            self.offset()
        self.buffer.append(trans)



class EpisodeReplayBuffer(object):

    def __init__(self, size):
        self.size = size
        self.buffer = []

    def get_single(self, index):
        return self.buffer[index]

    def offset(self):
        self.buffer.pop(0)
        # self.buffer = []

    def get_batch(self, batch_size):
        length = len(self.buffer)
        indices = np.random.choice(length, batch_size, replace=False)
        batch_buffer = []
        for i in indices:
            batch_buffer.extend(self.buffer[i])
        return batch_buffer

    def add_experience(self, episode):
        est_len = 1 + len(self.buffer)
        if est_len > self.size:
            self.offset()
        self.buffer.append(episode)
