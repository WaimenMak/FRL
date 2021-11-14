# -*- coding: utf-8 -*-
# @Time    : 2021/11/10 19:56
# @Author  : Weiming Mai
# @FileName: test_agent.py
# @Software: PyCharm

# import pynvml
import os
from copy import deepcopy

import gym
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import MongoObserver

from Utils.Tools import try_gpu, set_seed, FractionScheduler
# import random
from agents.DQN import fed_DQN
from models.Network import MLP

#save model
model_path = 'outputs/fed_model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


ex = Experiment("fed_dqn_single_test")  #name of the experiment
observer_mongo = MongoObserver(url='localhost:27017', db_name='DRL')
ex.observers.append(observer_mongo)
# ex.observers.append(MongoObserver.create(url='localhost:27017',
#                                          db_name='sacred'))


class Arguments():
    def __init__(self):
        pass

@ex.config
def Config():
    '''
    config parameters, send to mongodb
    :return: 
    '''
    args = Arguments()
    args.epsilon = 0.01
    args.local_bc = 64  #local update memory batch size
    args.local_epi = 10 # local update episode 最好是C_iter倍数
    args.gamma = 0.98
    args.lr = 0.002
    args.episode_num = 50
    args.eval_episode = 30
    args.capacity = 10000
    args.C_iter = 5
    args.train_seed = 1
    args.predict_seed = 10
    args.show_memory = False
    args.seed = 1
    args.env_name = 'CartPole-v1'
    args.frac = 1      # [0,1] choosing agents
    args.client_num = 1
    args.device = try_gpu()
    args.round = 400    #communication
    args.N = 10 # global update c_iter
    args.T = 1 #exploration
    args.E = 1 #update Q net
    args.B = 10 #sample data and update
    seed = 1
    #FedAvg Parameters

def eval(agent, env, eval_episode, name, log=True):
    r = 0
    for i_ep in range(eval_episode):
        state = env.reset()
        ep_reward = 0
        while True:
            # env.render()
            action = agent.predict(state)
            n_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = n_state
            if done == True:
                break
        # env.close()
        if (i_ep+1) % 30 == 0:
            print(f"{agent.name}, Episode:{i_ep+1}/{eval_episode}, Reward: {ep_reward}")
        if log:
            ex.log_scalar(name, ep_reward)
        r += ep_reward

    return r/eval_episode

@ex.automain
def main_eval(args):
    glob_env = gym.make(args.env_name)
    fresh_agent = fed_DQN(glob_env.observation_space.shape[0], glob_env.action_space.n, args.epsilon, args.local_bc, args.capacity, args.gamma, args.lr, args.device)

    fresh_agent.name = 'agent_server'
    fresh_agent.load(model_path + 'fedsgd.pth')

    eval(fresh_agent, glob_env, args.eval_episode, fresh_agent.name, log=True)