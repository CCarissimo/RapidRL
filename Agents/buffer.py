# Source: PhilTabor github SAC
import numpy as np
from random import sample


class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        size = min(self.size, batch_size)
        indices = sample(range(self.size), size-1)
        S = [self.buffer[index] for index in indices]
        S.append(self.buffer[self.index-1])
        return S

    def reset(self):
        self.buffer = [None] * self.max_size
        self.index = 0
        self.size = 0
