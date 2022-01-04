# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 14:46
# @Author  : Weiming Mai
# @FileName: Network.py
# @Software: PyCharm
import torch.nn.functional as F
import torch
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

class mlp_policy(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(mlp_policy, self).__init__()
        self.action_bound = action_bound
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        l1 = F.relu(self.fc1(x))
        l2 = F.relu(self.fc2(l1))
        action = self.action_bound * torch.tanh(self.fc3(l2))

        return action

    def R_t(self, x):
        l1 = F.relu(self.fc1(x))
        l2 = F.relu(self.fc2(l1))

        return l2

class mlp_value(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(mlp_value, self).__init__()
        self.q1_fc1 = nn.Linear(state_dim+action_dim, 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_fc3 = nn.Linear(256, 1)

        self.q2_fc1 = nn.Linear(state_dim+action_dim, 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_fc3 = nn.Linear(256, 1) #Q(s,a)

    def forward(self, s, a):

        x = torch.cat([s,a], dim=1)  # bc * input dim (state_dim + action_dim)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1_oupt = self.q1_fc3(q1)

        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2_oupt = self.q2_fc3(q2)

        return q1_oupt, q2_oupt

    def Q1_val(self, s, a):
        x = torch.cat([s, a], dim=1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1_oupt = self.q1_fc3(q1)

        return q1_oupt
    def R_t1(self, s, a):
        x = torch.cat([s, a], dim=1)
        q1 = F.relu(self.q1_fc1(x))
        rep = F.relu(self.q1_fc2(q1))

        return rep

    def R_t2(self, s, a):
        x = torch.cat([s, a], dim=1)
        q2 = F.relu(self.q2_fc1(x))
        rep = F.relu(self.q2_fc2(q2))

        return rep

