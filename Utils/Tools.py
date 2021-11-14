# -*- coding: utf-8 -*-
# @Time    : 2021/11/1 21:53
# @Author  : Weiming Mai
# @FileName: Tools.py
# @Software: PyCharm

import torch
import random
import numpy as np
import pynvml

class FractionScheduler:
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr / (1+num_update)


def try_gpu(): #single gpu
    i = 0
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


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
