# Source: PhilTabor github SAC
import numpy as np
from random import sample


# class ReplayBuffer():
#     def __init__(self, max_size):
#         self.mem_size = max_size
#         self.mem_cntr = 0
#         self.memory = []
#
#     def store_trajectory(self, transition):
#         self.memory.append(transition)
#         self.mem_cntr += 1
#         if self.mem_cntr > self.mem_size:
#             self.memory.popleft()
#
#     def sample_buffer(self, batch_size):
#         max_mem = min(self.mem_cntr, self.mem_size)
#
#         batch = np.random.choice(max_mem, batch_size)
#
#         return states, actions, rewards, states_, dones

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

    def unpack_sample(self, S):
        return S

    def sample(self, batch_size):
        indices = sample(range(self.size), batch_size)
        S = [self.buffer[index] for index in indices]
        S = self.unpack_sample(S)
        return S