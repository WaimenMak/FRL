# -*- coding: utf-8 -*-
# @Time    : 2022/5/4 19:45
# @Author  : Weiming Mai
# @FileName: Conv_model.py
# @Software: PyCharm

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class conv_policy(nn.Module):
    def __init__(self, inpt_chann, action_dim, action_bound):
        """
        input size 96*96*4, ouput size 
        :param inpt_chann:  
        :param oupt_chann: 
        :param action_dim: 
        :param action_bound: 
        """
        super(conv_policy, self).__init__()
        self.lower_bound = action_bound[0, :]
        self.upper_bound = action_bound[1, :]
        self.conv1 = nn.Conv2d(inpt_chann, 32, kernel_size=4, stride=4) # 24 * 24 * 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=3)         # 8 * 8 * 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)         # 4 * 4 *  128
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)         # 2 * 2 *  256
        self.oupt = nn.Sequential(nn.Flatten(),
                                  nn.Linear(2 * 2 * 256, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, action_dim))
        # self.fc1 = nn.Linear(9 * 9 * 64, 512)
        # self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255
        oupt = F.relu(self.conv1(x))
        oupt = F.relu(self.conv2(oupt))
        oupt = F.relu(self.conv3(oupt))
        oupt = F.relu(self.conv4(oupt))
        oupt = torch.tanh(self.oupt(oupt))  #[-1, 1]
        if oupt.device != self.lower_bound.device:
            self.lower_bound = self.lower_bound.to(oupt.device)
            self.upper_bound = self.upper_bound.to(oupt.device)

        oupt = torch.min(torch.max(oupt, self.lower_bound), self.upper_bound)

        return oupt


class distill_qnet(nn.Module):
    def __init__(self, inpt_chann, action_dim):
        super(distill_qnet, self).__init__()
        # print('conv')
        # self.feature_q1 = nn.Sequential(OrderedDict([('q1_l1', nn.Conv2d(inpt_chann, 32, kernel_size=8, stride=4)),
        #                                              ('relu1', nn.ReLU()),
        #                                              ('q1_l2', nn.Conv2d(32, 64, kernel_size=4, stride=2)),
        #                                              ('relu2', nn.ReLU()),
        #                                              ('q1_l3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
        #                                              ('q1_flat', nn.Flatten())]))
        self.feature_q1 = nn.Sequential(OrderedDict([('q1_l1', nn.Conv2d(inpt_chann, 32, kernel_size=4, stride=4)),
                                                     ('relu1', nn.ReLU()),
                                                     ('q1_l2', nn.Conv2d(32, 64, kernel_size=3, stride=3)),
                                                     ('relu2', nn.ReLU()),
                                                     ('q1_l3', nn.Conv2d(64, 128, kernel_size=2, stride=2)),
                                                     ('relu3', nn.ReLU()),
                                                     ('q1_l4', nn.Conv2d(128, 256, kernel_size=2, stride=2)), #[2, 2, 256]
                                                     ('relu4', nn.ReLU()),
                                                     ('q1_l5', nn.Conv2d(256, 256, kernel_size=2, stride=2)), #[1, 1, 256]
                                                     ('q1_flat', nn.Flatten())]))

        self.oupt_layer_q1 = nn.Sequential(nn.Linear(1 * 1 * 256 + action_dim, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, 1))


        self.feature_q2 = nn.Sequential(OrderedDict([('q2_l1', nn.Conv2d(inpt_chann, 32, kernel_size=4, stride=4)),
                                                     ('relu1', nn.ReLU()),
                                                     ('q2_l2', nn.Conv2d(32, 64, kernel_size=3, stride=3)),
                                                     ('relu2', nn.ReLU()),
                                                     ('q2_l3', nn.Conv2d(64, 128, kernel_size=2, stride=2)),
                                                     ('relu3', nn.ReLU()),
                                                     ('q2_l4', nn.Conv2d(128, 256, kernel_size=2, stride=2)),
                                                     ('relu4', nn.ReLU()),
                                                     ('q2_l5', nn.Conv2d(256, 256, kernel_size=2, stride=2)),
                                                     ('q2_flat', nn.Flatten())]))

        self.oupt_layer_q2 = nn.Sequential(nn.Linear(1 * 1 * 256 + action_dim, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, 1))
    def forward(self, s, a):

        # x = torch.cat([s,a], dim=1)  # bc * input dim (state_dim + action_dim)
        s = s / 255
        q1_rep = self.feature_q1(s)
        q1_oupt = self.oupt_layer_q1(torch.cat([q1_rep, a], dim=1))

        q2_rep = self.feature_q2(s)
        q2_oupt = self.oupt_layer_q2(torch.cat([q2_rep, a], dim=1))


        return q1_oupt, q2_oupt

    def Q1_val(self, s, a):
        s = s / 255
        q1_rep = self.feature_q1(s)
        q1_oupt = self.oupt_layer_q1(torch.cat([q1_rep, a], dim=1))

        return q1_oupt

    def R_t1(self, s, a):  #for moon
        # x = torch.cat([s, a], dim=1)
        s = s / 255
        rep = self.feature_q1(s)
        rep = self.oupt_layer_q1[0](torch.cat([rep, a], dim=1))

        return rep

    def R_t2(self, s, a):
        # x = torch.cat([s, a], dim=1)
        s = s / 255
        rep = self.feature_q2(s)
        rep = self.oupt_layer_q2[0](torch.cat([rep, a], dim=1))

        return rep

    def client_rep(self, s):  #layer 1 representation
        #output the representation of state, for center dataset
        s = s / 255
        rp1 = self.feature_q1(s)
        rp2 = self.feature_q2(s)
        # rp1 = F.relu(self.q1_fc1(x))
        # rp2 = F.relu(self.q2_fc1(x))

        return rp1, rp2

    def server_oupt(self, rp1, rp2):
        #input are the mean features of s concated a, for the label in center dataset
        q1_oupt = self.oupt_layer_q1(rp1)

        q2_oupt = self.oupt_layer_q2(rp2)

        return q1_oupt, q2_oupt

    def shared_params(self):
        #share parameters to the server
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


