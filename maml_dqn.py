# -*- coding: utf-8 -*-
# @Time    : 2021/11/4 16:47
# @Author  : Weiming Mai
# @FileName: maml_dqn.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 2021/11/1 14:45
# @Author  : Weiming Mai
# @FileName: fedavg_dqn.py
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
from agents.DQN import DQN
from models.Network import MLP

#save model
model_path = 'outputs/maml_model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


ex = Experiment("maml_dqn")  #name of the experiment
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
    args.local_epi = 15 # local update episode 最好是C_iter倍数
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
    args.env_name = 'CartPole-v0'
    args.frac = 1      # [0,1] choosing agents
    args.client_num = 3
    args.device = try_gpu()
    args.round = 3    #communication
    seed = 1
    #FedAvg Parameters

#environment setting
def agent_env_config(args, env_name):
    env = gym.make(env_name)
    # env.seed(args.train_seed)
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2
    agent = DQN(state_dim, action_dim, args.epsilon, args.local_bc, args.capacity, args.gamma, args.lr, args.device)
    return agent, env

def eval(agent, env, eval_episode, name, log=True):
    ep_reward = 0
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
        if (i_ep+1) % 5 == 0:
            print(f"Episode:{i_ep+1}/{eval_episode}, Reward: {ep_reward}")
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
        agent, local_env = agent_env_config(args, env_name='CartPole-v0')
        agents.append(agent)
        local_envs.append(local_env)

    return agents, local_envs


def ClientUpdate(args, agent, k, params, local_env):
    '''
    :param agent: dqn class
    :param params: global net state dict
    :param args: hyper parameters
    :k : kth agent
    :return: 
    '''
    agent.target_net.load_state_dict(params)
    agent.policy_net.load_state_dict(agent.target_net.state_dict())
    r = 0
    for i_ep in range(args.local_epi):
        state = local_env.reset()
        ep_reward = 0
        while True:
            action = agent.choose_action(state)
            n_state, reward, done, _ = local_env.step(action)
            ep_reward += reward
            agent.memory.add(state, action, reward, n_state, done)
            agent.UpdateQ()
            state = n_state
            if done == True:
                break
        if (i_ep + 1) % args.C_iter == 0:
            agent.UpdateTarget()

        print(f"Agent {k}: Episode:{i_ep+1}/{args.local_epi}, Reward: {ep_reward}")
        ex.log_scalar("agent_"+str(k)+"_train_ep_reward", ep_reward)
        r += eval(agent, local_env, eval_episode=1, name='agent_'+str(k)+'_eval_ep_reward')

    return r/args.local_epi

def ServerUpdate(agents, weighted): #FedAvg
    global_net = deepcopy(agents[0].target_net.state_dict())
    with torch.no_grad():
        K = len(agents)
        for params in global_net.keys():
            global_net[params] = weighted[0] * agents[0].target_net.state_dict()[params]
        for params in global_net.keys():
            for k in range(1, K):
                global_net[params] += weighted[k] * agents[k].target_net.state_dict()[params]
            # global_net[params] = torch.div(global_net[params], K)
    return global_net

def test_server_agent(local_envs, agent, eval_episode):
    k = 0
    for env in local_envs:
        eval(agent, env, eval_episode, 'fresh_test_env' + str(k))
        k += 1



@ex.automain
def main(args, seed):
    # set_seed(args.train_seed)
    env = gym.make(args.env_name)
    set_seed(seed)
    global_model = MLP(env.observation_space.shape[0], env.action_space.n).to(args.device)
    net_glob = global_model.state_dict()
    fresh_agent = DQN(env.observation_space.shape[0], env.action_space.n, args.epsilon, args.local_bc, args.capacity, args.gamma, args.lr, args.device)
    agents, local_envs = GenerateAgent(args)
    # choose number of clients
    m = max(int(args.frac * args.client_num), 1)
    print("Begin Train:")
    #communications time
    for round in range(args.round):
        weighted = np.zeros(args.client_num)
        idxs_clients = np.random.choice(range(args.client_num), m, replace=False)
        #this part should be parallel
        for idx in idxs_clients:
            weighted[idx] = ClientUpdate(args, agents[idx], idx, net_glob, local_envs[idx])
        # weighted = list(map(lambda x: x/sum(weighted), weighted))
        weighted = weighted/np.sum(weighted)
        net_glob = ServerUpdate(agents, weighted)
        fresh_agent.policy_net.load_state_dict(net_glob)
        mean_r = eval(fresh_agent, env, args.eval_episode, name=None, log=False)
        ex.log_scalar('Round_test_ep_reward', mean_r, round)

    torch.save(net_glob, model_path + 'fed_dqn.pth')
    print("Begin Test:")
    # fresh_agent = DQN(env.observation_space.shape[0], env.action_space.n, args.epsilon, args.local_bc, args.capacity, args.gamma, args.lr, args.device)
    fresh_agent.load(model_path)
    #Testing
    # agent = DQN(state_dim, action_dim, args.epsilon, args.batch_size, args.capacity, args.gamma, args.lr, args.device)
    # agent.load(model_path)
    eval(fresh_agent, env, args.eval_episode, name='fresh_test_ep_reward')
    eval(agents[0], env, args.eval_episode, name='agent_0_test_ep_reward')
    eval(agents[1], env, args.eval_episode, name='agent_1_test_ep_reward')
    eval(agents[2], env, args.eval_episode, name='agent_2_test_ep_reward')
    test_server_agent(local_envs, fresh_agent, args.eval_episode)