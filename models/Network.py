# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 14:46
# @Author  : Weiming Mai
# @FileName: Network.py
# @Software: PyCharm
import torch.nn.functional as F
import torch
from torch import nn
from collections import OrderedDict

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


class distill_qnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(distill_qnet, self).__init__()
        print('debug')
        self.feature_q1 = nn.Sequential(
            OrderedDict([('q1_l1', nn.Linear(state_dim + action_dim, 256)), ('relu1', nn.ReLU()), ('q1_l2', nn.Linear(256, 256)), ('relu2', nn.ReLU())])
        )
        # self.feature_q1 = nn.Sequential(
        #     OrderedDict([('q1_l1', nn.Linear(state_dim + action_dim, 256)), ('relu1', nn.ReLU())])
        # )
        self.oupt_layer_q1 = nn.Sequential(
            OrderedDict([('q1_l3',nn.Linear(256, 256)), ('relu3', nn.ReLU()), ('q1_l4', nn.Linear(256, 1))])
        )

        self.feature_q2 = nn.Sequential(
            OrderedDict([('q2_l1', nn.Linear(state_dim + action_dim, 256)), ('relu1', nn.ReLU()), ('q2_l2', nn.Linear(256, 256)), ('relu2', nn.ReLU())])
        )
        # self.feature_q2 = nn.Sequential(
        #     OrderedDict([('q2_l1', nn.Linear(state_dim + action_dim, 256)), ('relu1', nn.ReLU())])
        # )
        self.oupt_layer_q2 = nn.Sequential(
            OrderedDict([('q2_l3',nn.Linear(256, 256)), ('relu3', nn.ReLU()), ('q2_l4', nn.Linear(256, 1))])
        )

    def forward(self, s, a):

        x = torch.cat([s,a], dim=1)  # bc * input dim (state_dim + action_dim)
        q1_rep = self.feature_q1(x)
        q1_oupt = self.oupt_layer_q1(q1_rep)

        q2_rep = self.feature_q2(x)
        q2_oupt = self.oupt_layer_q2(q2_rep)


        return q1_oupt, q2_oupt

    def Q1_val(self, s, a):
        x = torch.cat([s, a], dim=1)
        q1_rep = self.feature_q1(x)
        q1_oupt = self.oupt_layer_q1(q1_rep)

        return q1_oupt

    def R_t1(self, s, a):  #for moon
        x = torch.cat([s, a], dim=1)
        rep = self.feature_q1(x)
        l = len(self.oupt_layer_q1)
        for i in range(l):
            rep = self.oupt_layer_q1[i](rep)
            if i == l - 2:   #layer before output
                break
        # q1 = F.relu(self.q1_fc1(x))
        # q1 = F.relu(self.q1_fc2(q1))
        # rep = F.relu(self.q1_fc3(q1))

        return rep

    def R_t2(self, s, a):
        x = torch.cat([s, a], dim=1)
        rep = self.feature_q2(x)
        l = len(self.oupt_layer_q2)
        for i in range(l):
            rep = self.oupt_layer_q2[i](rep)
            if i == l - 2:
                break
        # q2 = F.relu(self.q2_fc1(x))
        # q2 = F.relu(self.q2_fc2(q2))
        # rep = F.relu(self.q2_fc3(q2))

        return rep

    def client_rep(self, s, a):  #layer 1 representation
        x = torch.cat([s, a], dim=1)
        rp1 = self.feature_q1(x)
        rp2 = self.feature_q2(x)
        # rp1 = F.relu(self.q1_fc1(x))
        # rp2 = F.relu(self.q2_fc1(x))

        return rp1, rp2

    def server_oupt(self, rp1, rp2):
        q1_oupt = self.oupt_layer_q1(rp1)
        # q1 = F.relu(self.q1_fc2(rp1))
        # q1 = F.relu(self.q1_fc3(q1))
        # q1_oupt = self.q1_fc4(q1)

        q2_oupt = self.oupt_layer_q2(rp2)
        # q2 = F.relu(self.q2_fc2(rp2))
        # q2 = F.relu(self.q2_fc3(q2))
        # q2_oupt = self.q2_fc4(q2)

        return q1_oupt, q2_oupt

    def shared_params(self):
        q1_params = self.oupt_layer_q1.state_dict()
        q2_params = self.oupt_layer_q2.state_dict()

        return q1_params, q2_params

    def client_update(self, glob_params):  #glob_params is a tuple of two state_dict
        self.oupt_layer_q1.load_state_dict(glob_params[0])
        self.oupt_layer_q2.load_state_dict(glob_params[1])
        # for param in self.oupt_layer_q1.state_dict().keys():
        #     self.oupt_layer_q1.state_dict()[param].copy_(glob_params[0][param])
        # for param in self.oupt_layer_q1.state_dict().keys():
        #     self.oupt_layer_q1.state_dict()[param].copy_(glob_params[0][param])

    def server_update(self, local_q, glob_params):
        #for glob q updating
        self.oupt_layer_q1.load_state_dict(glob_params[0])
        self.oupt_layer_q2.load_state_dict(glob_params[1])

        self.feature_q1.load_state_dict(local_q.feature_q1.state_dict())
        self.feature_q2.load_state_dict(local_q.feature_q2.state_dict())