# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 14:46
# @Author  : Weiming Mai
# @FileName: Memory.py
# @Software: PyCharm
import random
from collections import deque

class replay_buffer():
    def __init__(self, capacity):
        self.buffer = deque()
        self.capacity = capacity
        self.count = 0

    def add(self, state, action, reward, n_state, done):  # done: whether the final state, TD error would be different.
        experience = (state, action, reward, n_state, done)
        if self.count < self.capacity:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample(self, batch_size):  # return a tuple
        batch = random.sample(self.buffer, batch_size)  # a list [(s,a,r,s), ...]
        return zip(*batch)

    #         return batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def __len__(self):
        # return len(self.buffer)
        return self.count