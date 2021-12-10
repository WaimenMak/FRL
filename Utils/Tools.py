# -*- coding: utf-8 -*-
# @Time    : 2021/11/1 21:53
# @Author  : Weiming Mai
# @FileName: Tools.py
# @Software: PyCharm

import torch
import random
import math
import numpy as np
import pynvml

class FractionScheduler:
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr / (1+num_update)

class ExponentialScheduler:
    def __init__(self, lr):
        self.lr_start = lr
        self.lr_end = 0.001
        self.lr_decay = 50

    def __call__(self, num_update):
        return self.lr_end + (self.lr_start - self.lr_end) * math.exp(- num_update/self.lr_decay)

def try_gpu(): #single gpu
    i = 0
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def action_trans(action, action_dim, upperbound, lowerbound):
    """
    discrete action to continuous action
    :param action_dim: 
    :return: 
    """
    conti_action = lowerbound + (action / (action_dim - 1)) * (upperbound - lowerbound)
    return conti_action

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def print_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  #GPU 0
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info
