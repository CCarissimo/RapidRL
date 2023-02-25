# Source: PhilTabor github SAC
import numpy as np
from random import sample


class ReplayMemory:
    def __init__(self, max_size, len_death_memories):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0
        self.death_memories = []
        self.len_death_memories = len_death_memories

    def append(self, obj):
        self.buffer[self.index] = obj
        if obj.terminal:
            for i in range(0, min(self.size, self.len_death_memories)):
                ind = (self.index - i) % self.max_size
                self.death_memories.append(self.buffer[ind])
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

        for obj in self.death_memories:  # append them to buffer
            self.buffer[self.index] = obj
            self.size = min(self.size + 1, self.max_size)
            self.index = (self.index + 1) % self.max_size

        # if not keep_deaths:
        #     tmp = []
        #     if size_stories > 0:
        #         sample = self.sample(size_stories)
        #         for s in sample:
        #             tmp.append(s)
        #     self.buffer = [None] * self.max_size
        #     self.index = 0
        #     self.size = 0
        #     for i in range(len(tmp)):
        #         self.append(tmp.pop())
        # else:
        #     tmp = []
        #     for i in range(len(self.buffer)):
        #         transition = self.buffer[i]
        #         if transition is None:
        #             continue
        #         if transition.terminal:
        #             tmp.append(transition)
        #     space = size_stories - len(tmp)
        #     if space > 0:
        #         sample = self.sample(space)
        #         for s in sample:
        #             tmp.append(s)
        #     self.buffer = [None] * self.max_size
        #     self.index = 0
        #     self.size = 0
        #     for i in range(len(tmp)):
        #         self.append(tmp.pop())
