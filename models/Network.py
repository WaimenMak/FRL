# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 14:46
# @Author  : Weiming Mai
# @FileName: Network.py
# @Software: PyCharm
import torch.nn.functional as F
from torch import nn

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        l1 = F.relu(self.fc1(x))
        l2 = F.relu(self.fc2(l1))
        output = self.fc3(l2)

        return output




