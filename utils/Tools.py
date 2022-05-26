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
    def __init__(self, lr_start, lr_end):
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_decay = 50

    def __call__(self, num_update):
        return self.lr_end + (self.lr_start - self.lr_end) * math.exp(- num_update/self.lr_decay)

def try_gpu(): #single gpu
    i = 0
    if torch.cuda.device_count() == i + 1:
        return torch.device(f'cuda:{i}')
    elif torch.cuda.device_count() == i + 2:
        return torch.device(f'cuda:{i+1}')
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


def _eval(agent, envs, args):

    env_num = 0
    std_list = [] #over all std
    mean_list = []
    each_env = []
    total = []
    for env in envs:
        env_num += 1
        r = 0
        for i_ep in range(args.test_episode):
            state = env.reset()
            if args.noisy_input:
                state = state + np.random.normal(env.mean, 0.01, state.shape[0])
            ep_reward = 0
            for iter in range(args.episode_length):
                action = agent.predict(state)  # action is array
                n_state, reward, done, _ = env.step(action)  # env.step accept array or list
                if args.noisy_input:
                    n_state = n_state + np.random.normal(env.mean, 0.01, state.shape[0])
                ep_reward += reward
                if done == True:
                    break
                state = n_state
            r += ep_reward / args.test_episode
            # temp += ep_reward/args.eval_episode
            each_env.append(ep_reward)

            total.append(ep_reward)
        # each_env.append(temp)
        mean = np.mean(each_env)
        std = np.std(each_env)

        std_list.append(mean)
        mean_list.append(f"{mean:.2f}+-{std:.2f}")
        print(f"env{env_num}:mean {mean:.2f}, std {std:.2f}", end=" ")
        each_env.clear()

    # mean_list.append(f"{np.mean(total):.2f}+-{np.std(std_list):.2f}")
    print(f"overall mean:{np.mean(total):.2f}, std {np.std(std_list):.2f}")
    return np.mean(total)

def _test(agent, local_envs, args):
    overall = _eval(agent, local_envs, args)
    # overall = np.mean(res)
    print(f"{overall:.2f}")



