# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 14:46
# @Author  : Weiming Mai
# @FileName: Memory.py
# @Software: PyCharm
import random
from collections import deque
from torch.utils.data import Dataset
import pickle
import torch

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

    def distil_sample(self, batch_size, epc):  # return a tuple
        if self.count - epc * batch_size > 0:
            batch = random.sample(list(self.buffer)[(self.count - epc * batch_size):], batch_size)  # a list [(s,a,r,s), ...]
        else:
            batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def save(self, PATH):
        pickle.dump(self.buffer, open(PATH, "wb"))

    def load(self, PATH):
        with open(PATH, "rb") as f:
            self.buffer = pickle.loads(f.read())

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def __len__(self):
        # return len(self.buffer)
        return self.count

class DistilDataset(Dataset):
    def __init__(self):
        self.tensors = [torch.tensor([1]) for _ in range(4)]   #initialize as 1

    def __getitem__(self, item):
        rep1 = self.tensors[0][item]
        rep2 = self.tensors[1][item]
        label1 = self.tensors[2][item]
        label2 = self.tensors[3][item]

        return rep1, rep2, label1, label2

    def __len__(self):
        return self.tensors[0].size(0)

    def clear(self):
        del self.tensors
        self.tensors = [torch.tensor([]) for _ in range(4)]

    def add(self, tensors):
        for i in range(len(tensors)):
            self.tensors[i] = torch.cat((self.tensors[i], tensors[i]))

        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
