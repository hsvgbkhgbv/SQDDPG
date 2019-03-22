import numpy as np


class ReplayBuffer(object):

    def __init__(self, size, forget_rate):
        self.size = size
        self.buffer = []
        self.forget_rate = forget_rate

    def get_episode(self, index):
        return self.buffer[index]

    def offset(self):
        self.buffer = self.buffer[int(self.size*self.forget_rate):]

    def get_batch_episodes(self, batch_size):
        length = len(self.buffer)
        indices = np.random.choice(length, batch_size)
        batch_buffer = [self.buffer[i] for i in indices]
        return batch_buffer

    def add_experience(self, batch_data):
        est_len = len(batch_data) + len(self.buffer)
        if est_len > self.size:
            self.offset()
        self.buffer.extend(batch_data)
