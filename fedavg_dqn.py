# -*- coding: utf-8 -*-
# @Time    : 2021/11/4 16:47
# @Author  : Weiming Mai
# @FileName: maml_dqn.py
# @Software: PyCharm


# import pynvml
import os
from copy import deepcopy

import gym
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import MongoObserver

from Utils.Tools import try_gpu, set_seed
# import random
from agents.DQN import fed_DQN
from models.Network import MLP

#save model
model_path = 'outputs/fed_model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


ex = Experiment("fed_dqn")  #name of the experiment
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
    args.client_num = 8
    args.device = try_gpu()
    args.round = 300    #communication
    args.N = 50 # global update
    args.T = 10 #exploration
    args.E = 1 #update Q net
    seed = 1
    #FedAvg Parameters

#environment setting
def agent_env_config(args, env_name):
    env = gym.make(env_name)
    # env.seed(args.train_seed)
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2
    agent = fed_DQN(state_dim, action_dim, args.epsilon, args.local_bc, args.capacity, args.gamma, args.lr, args.device)
    return agent, env

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


def GenerateAgent(args):
    '''
    :param args: 
    :return: local agents and local envs
    '''
    agents = []
    local_envs = []
    for i in range(args.client_num):
        agent, local_env = agent_env_config(args, env_name='CartPole-v1')
        agent.name = 'agent' + str(i)
        agents.append(agent)
        local_envs.append(local_env)

    return agents, local_envs

def Exploration(agent, local_env, T, state_info):
    '''
    :param agent: 
    :param local_env: 
    :param T: exploration time
    :param state_info: 
    :return: 
    '''
    if state_info[0] == True:
        state = local_env.reset()
    else:
        state = state_info[1]
    t = 0
    while True:
        if t != 0:
            state = local_env.reset()
        # ep_reward = 0
        while True:
            action = agent.choose_action(state)
            n_state, reward, done, _ = local_env.step(action)
            # ep_reward += reward
            agent.memory.add(state, action, reward, n_state, done)
            t += 1
            state = n_state
            if done == True or t == T:
                break
        if t == T:
            break
    return done, n_state

def Exploration_2(agent, local_env, T):
    '''
    :param agent: 
    :param local_env: 
    :param T: exploration time
    :param state_info: 
    :return: 
    '''
    for i_ep in range(T):
        state = local_env.reset()
        # ep_reward = 0
        while True:
            action = agent.choose_action(state)
            n_state, reward, done, _ = local_env.step(action)
            # ep_reward += reward
            agent.memory.add(state, action, reward, n_state, done)
            state = n_state
            if done == True:
                break

def ClientUpdate(args, agent, k, params, local_env, n, state_info):
    '''
    :param agent: dqn class
    :param params: global net state dict
    :param args: hyper parameters
    :k : kth agent
    :n : global model update times
    :return: 
    '''
    if n % args.N == 0:
        agent.target_net.load_state_dict(params)
    agent.policy_net.load_state_dict(params)
    Exploration_2(agent, local_env, args.T)
    for i in range(args.E):
        # state_info = Exploration(agent, local_env, args.T, state_info)
        # for t in range(1):
        agent.UpdateQ()
        # if (i + 1) % args.N == 0:
        #     #update from server
        #     agent.UpdateTarget()
        # print(f"Agent {k}: Episode:{i_ep+1}/{args.local_epi}, Reward: {ep_reward}")
        # ex.log_scalar("agent_"+str(k)+"_train_ep_reward", ep_reward)
        eval(agent, local_env, eval_episode=3, name='agent_'+str(k)+'_eval_ep_reward')
    return state_info


def ServerUpdate(agents, weighted): #FedAvg
    global_net = deepcopy(agents[0].policy_net.state_dict())
    with torch.no_grad():
        K = len(agents)
        for params in global_net.keys():
            global_net[params] = weighted[0] * agents[0].policy_net.state_dict()[params]
        for params in global_net.keys():
            for k in range(1, K):
                global_net[params] += weighted[k] * agents[k].policy_net.state_dict()[params]
            # global_net[params] = torch.div(global_net[params], K)
    return global_net

def eval_agents(local_envs, agents, eval_episode, fresh_agent, glob_env):
    k = 0
    total_agents = deepcopy(agents)
    total_envs = deepcopy(local_envs)
    total_agents.append(fresh_agent)
    total_envs.append(glob_env)
    for env in total_envs:
        for agent in total_agents:
            eval(agent, env, eval_episode, agent.name+'_test_env' + str(k)) #'agent0_test_env0'
        k += 1



@ex.automain
def main(args, seed):
    set_seed(args.train_seed)
    glob_env = gym.make(args.env_name)
    set_seed(seed)
    global_model = MLP(glob_env.observation_space.shape[0], glob_env.action_space.n).to(args.device)
    net_glob = global_model.state_dict()
    fresh_agent = fed_DQN(glob_env.observation_space.shape[0], glob_env.action_space.n, args.epsilon, args.local_bc, args.capacity, args.gamma, args.lr, args.device)
    fresh_agent.name = 'agent_server'
    agents, local_envs = GenerateAgent(args)
    # choose number of clients
    m = max(int(args.frac * args.client_num), 1)
    print("Begin Train:")
    #communications time
    weighted = [agent.batch_size for agent in agents]
    weighted = list(map(lambda x: x / sum(weighted), weighted))
    state_info = [(True, 1) for i in range(args.client_num)]
    for round in range(args.round):

        idxs_clients = np.random.choice(range(args.client_num), m, replace=False)
        #this part should be parallel
        for idx in idxs_clients:
            state_info[idx] = ClientUpdate(args, agents[idx], idx, net_glob, local_envs[idx], round, state_info[idx])
        # weighted = list(map(lambda x: x/sum(weighted), weighted))
        net_glob = ServerUpdate(agents, weighted)
        fresh_agent.policy_net.load_state_dict(net_glob)
        mean_r = eval(fresh_agent, glob_env, args.eval_episode, name=None, log=False)
        ex.log_scalar('Round_test_ep_reward', mean_r, round)

    torch.save(net_glob, model_path + 'fed_dqn.pth')
    print("Begin Test:")
    # fresh_agent = DQN(env.observation_space.shape[0], env.action_space.n, args.epsilon, args.local_bc, args.capacity, args.gamma, args.lr, args.device)
    fresh_agent.load(model_path)


    eval_agents(local_envs, agents, args.eval_episode, fresh_agent, glob_env)
