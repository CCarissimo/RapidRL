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

    def reset(self, size_stories, keep_deaths=False):
        if not keep_deaths:
            tmp = []
            if size_stories > 0:
                sample = self.sample(size_stories)
                for s in sample:
                    tmp.append(s)
            self.buffer = [None] * self.max_size
            self.index = 0
            self.size = 0
            for i in range(len(tmp)):
                self.append(tmp.pop())
        else:
            tmp = []
            for i in range(len(self.buffer)):
                transition = self.buffer[i]
                if transition is None:
                    continue
                if transition.terminal:
                    tmp.append(transition)
            space = size_stories - len(tmp)
            if space > 0:
                sample = self.sample(space)
                for s in sample:
                    tmp.append(s)
            self.buffer = [None] * self.max_size
            self.index = 0
            self.size = 0
            for i in range(len(tmp)):
                self.append(tmp.pop())
