# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 14:42
# @Author  : Weiming Mai
# @FileName: DQN.py
# @Software: PyCharm

import os
import random

import gym
import numpy as np
import pynvml
import torch
from sacred import Experiment
from sacred.observers import MongoObserver

from agents.DQN import DQN

#save model
model_path = 'outputs/model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


ex = Experiment("dqn_Cartpole")  #name of the experiment
observer_mongo = MongoObserver(url='localhost:27017', db_name='DRL')
ex.observers.append(observer_mongo)
# ex.observers.append(MongoObserver.create(url='localhost:27017',
#                                          db_name='sacred'))

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

@ex.config
def Config():
    '''
    config parameters, send to mongodb
    :return: 
    '''
    epsilon = 0.01
    batch_size = 64
    gamma = 0.98
    lr = 0.002
    episode_num = 200
    eval_episode = 30
    capacity = 10000
    C_iter = 5
    train_seed = 1
    predict_seed = 10
    show_memory = False
    seed = 1

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--epsilon', type=float, default=0.01, help="epsilon")
    # parser.add_argument('--batch_size', type=int, default=64, help="train batch size")
    # parser.add_argument('--gamma', type=float, default=0.98, help="gamma")
    # parser.add_argument('--lr', type=float, default=0.002, help="learning rate")
    # parser.add_argument('--episode_num', type=int, default=200, help="train episode")
    # parser.add_argument('--eval_episode', type=int, default=30, help="eval episode")
    # parser.add_argument('--capacity', type=int, default=10000, help="memory capacity")
    # parser.add_argument('--C_iter', type=int, default=5, help="update target net")
    # parser.add_argument('--train_seed', type=int, default=1, help="train seed")
    # parser.add_argument('--predict_seed', type=int, default=10, help="predict seed")
    # parser.add_argument('--show_memory', action = 'store_true', help="print memory")
    # parser.add_argument('--seed', type=int, default=1, help="seed")
    # args = parser.parse_args()

def eval(agent, env, eval_episode, name):

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
        if (i_ep+1) % 2 == 0:
            print(f"Episode:{i_ep+1}/{eval_episode}, Reward: {ep_reward}")

        ex.log_scalar(name, ep_reward)

def train(agent, env, epsilon, batch_size, gamma, lr, episode_num, capacity, C_iter, memory):
    return_list = []
    for i_ep in range(episode_num):
        state = env.reset()
        done = False
        ep_reward = 0
        #     print(state)
        while True:
            action = agent.choose_action(state)
            n_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.add(state, action, reward, n_state, done)
            agent.UpdateQ()
            state = n_state
            #         print(n_state, action)
            if done == True:
                break
        if (i_ep+1) % C_iter == 0:
            agent.UpdateTarget()

        if (i_ep+1) % 10 == 0:
            if (memory):
                print(f"Episode:{i_ep+1}/{episode_num}, Reward: {ep_reward}, Memory: {memory.used/ 1024 / 1024:.1f} MB/{memory.total/ 1024 / 1024:.1f} MB")
            else:
                print(f"Episode:{i_ep+1}/{episode_num}, Reward: {ep_reward}")
        return_list.append(ep_reward)

        eval(agent, env, eval_episode=1, name = 'eval_ep_reward')
        ex.log_scalar("train_ep_reward", ep_reward, i_ep)

#environment setting
def env_config(train_seed):
    env = gym.make('CartPole-v0')
    env.seed(train_seed)
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2
    return state_dim, action_dim, env

@ex.automain
def main(epsilon, batch_size, gamma, lr, episode_num, eval_episode, capacity, C_iter, train_seed, predict_seed, show_memory):
    #show the info of the memory of GPU
    memory = None
    state_dim, action_dim, env = env_config(train_seed)
    device = try_gpu()
    set_seed(train_seed)
    agent = DQN(state_dim, action_dim, epsilon, batch_size, capacity, gamma, lr, device)
    print("Begin train:")
    if show_memory == True:
        memory = print_memory()
    train(agent, env, epsilon, batch_size, gamma, lr, episode_num, capacity, C_iter, memory)
    agent.save(model_path)
    #Testing
    print("Test:")
    state_dim, action_dim, env = env_config(predict_seed)
    agent = DQN(state_dim, action_dim, epsilon, batch_size, capacity, gamma, lr, device)
    agent.load(model_path)
    eval(agent, env, eval_episode, name = 'test_ep_reward')
