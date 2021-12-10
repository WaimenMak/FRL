# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 17:07
# @Author  : Weiming Mai
# @FileName: fed_dqn_pendulum.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 2021/11/4 16:47
# @Author  : Weiming Mai
# @FileName: maml_dqn.py
# @Software: PyCharm



import sys
import os
if __name__ == '__main__':
    sys.path.append(os.path.dirname(sys.path[0]))

from copy import deepcopy

import gym
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import MongoObserver
from non_stationary_envs.Pendulum import PendulumEnv, pendulum_env_config
from utils.Tools import try_gpu, set_seed, ExponentialScheduler, action_trans
# import random
from agents.DQN import fed_DQN
from models.Network import MLP

#save model
model_path = '../outputs/fed_model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


ex = Experiment("fedavgdqn_10_envs")  #name of the experiment
observer_mongo = MongoObserver(url='localhost:27017', db_name='DRL')
ex.observers.append(observer_mongo)


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
    args.eps_decay = None
    args.local_bc = 128  #local update memory batch size
    # args.local_epi = 10 # local update episode 最好是C_iter倍数
    args.gamma = 0.98
    args.lr = 0.01
    # args.episode_num = 50
    args.eval_episode = 2
    args.episode_length = 200
    args.capacity = 10000
    # args.C_iter = 5
    args.train_seed = 0
    args.env_seed = None
    # args.predict_seed = 10
    args.show_memory = False
    # args.seed = 1
    # args.env_name = 'Pendulum_normal'
    args.env_name = 'Pendulum'
    args.action_dim = 11
    args.frac = 1      # [0,1] choosing agents
    args.client_num = 5
    args.device = try_gpu()
    args.round = 200    #communication
    args.N = 5 # global update round
    args.N_decay = False
    args.scheduler = True
    args.T = 5 #exploration
    args.E = 1 #update Q net
    args.B = 30 #sample data and update
    args.filename = "_".join(('clients', str(args.client_num), args.env_name, 'fedavgdqn.pth'))
    seed = 1
    #FedAvg Parameters

def eval(length, agent, env, eval_episode, name, log=True):
    r = 0
    upperbound = env.action_space.low[0]
    lowerbound = env.action_space.high[0]
    for i_ep in range(eval_episode):
        state = env.reset()
        ep_reward = 0
        for _ in range(length):
            # env.render()
            action = agent.predict(state)
            action = action_trans(action, agent.action_dim, upperbound, lowerbound)
            n_state, reward, done, _ = env.step([action])
            ep_reward += reward
            state = n_state
            if done == True:
                break
        # env.close()
        # if (i_ep+1) % 30 == 0:
        #     print(f"{agent.name}, Episode:{i_ep+1}/{eval_episode}, Reward: {ep_reward}")
        if log:
            ex.log_scalar(name, ep_reward)  # just for eval agent in local envs
        r += ep_reward

    return r/eval_episode

#environment setting
def agent_env_config(args, seed, env_name):
    if env_name == 'Pendulum_normal':
        # env = gym.make(env_name)
        env = PendulumEnv()
    else:
        env = pendulum_env_config(seed) # seed
    # env.seed(args.train_seed)
    state_dim = env.observation_space.shape[0]  # 3
    # action_dim = env.action_space.n  # 15
    agent = fed_DQN(state_dim, args.action_dim, args)
    return agent, env

def GenerateAgent(args):
    '''
    :param args: 
    :return: local agents and local envs
    '''
    agents = []
    local_envs = []
    for i in range(args.client_num):
        agent, local_env = agent_env_config(args, seed=i, env_name=args.env_name)
        agent.name = 'agent' + str(i)
        agents.append(agent)
        local_envs.append(local_env)

    return agents, local_envs

def Exploration_2(args, agent, local_env, T):
    '''
    :param agent: 
    :param local_env: 
    :param T: exploration time
    :param state_info: 
    :return: 
    '''
    upperbound = local_env.action_space.low[0]
    lowerbound = local_env.action_space.high[0]
    for i_ep in range(T):
        state = local_env.reset()
        # ep_reward = 0
        for _ in range(args.episode_length):
            discrete_action = agent.choose_action(state)
            action = action_trans(discrete_action, agent.action_dim, upperbound, lowerbound)
            n_state, reward, done, _ = local_env.step([action])
            # ep_reward += reward
            agent.memory.add(state, discrete_action, reward, n_state, done)
            state = n_state
            if done == True:
                break

def ClientUpdate(args, agent, k, params, local_env, n, scheduler):
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
    while len(agent.memory) < args.local_bc:
        Exploration_2(args, agent, local_env, 5)
    Exploration_2(args, agent, local_env, args.T)
    for i in range(args.E):
        # state_info = Exploration(agent, local_env, args.T, state_info)
        Exploration_2(args, agent, local_env, args.T)
        for b in range(args.B):
            if scheduler:
                #with learning rate decay
                agent.UpdateQ(Round=n, scheduler=scheduler)
            else:
                #without learning rate decay
                agent.UpdateQ(ddqn=True)
        # eval each agent after local update
        eval(args.episode_length, agent, local_env, eval_episode=1, name='agent_'+str(k)+'_eval_ep_reward')
    # return state_info


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

def eval_agents(args, local_envs, agents, eval_episode, fresh_agent, glob_env):
    k = 0
    total_agents = deepcopy(agents)
    total_envs = deepcopy(local_envs)
    total_agents.append(fresh_agent)
    total_envs.append(glob_env)
    for env in total_envs:
        for agent in total_agents:
            R = eval(args.episode_length, agent, env, eval_episode, agent.name+'_test_env' + str(k)) #'agent0_test_env0'
            print(f"{agent.name}, Env:{k}/{len(local_envs)-1}, Reward: {R:.2f}")
        k += 1



@ex.automain
def main(args, seed):
    set_seed(args.train_seed)
    glob_env = pendulum_env_config(args.env_seed)
    set_seed(seed)
    scheduler = None
    if args.scheduler:
        scheduler = ExponentialScheduler(args.lr)

    global_model = MLP(glob_env.observation_space.shape[0], args.action_dim).to(args.device)
    net_glob = global_model.state_dict()
    fresh_agent = fed_DQN(glob_env.observation_space.shape[0], args.action_dim, args)
    fresh_agent.name = 'agent_server'
    agents, local_envs = GenerateAgent(args)
    # choose number of clients
    m = max(int(args.frac * args.client_num), 1)
    print("Begin Train:")
    #communications time
    weighted = [agent.batch_size for agent in agents]
    weighted = list(map(lambda x: x / sum(weighted), weighted))
    # state_info = [(True, 1) for i in range(args.client_num)]
    for round_ in range(args.round):
        if args.N_decay:
            if round_ > int(args.round/3):
                args.N = args.N * 2
            elif round_ > int(args.round*2/3):
                args.N = args.N * 3

        idxs_clients = np.random.choice(range(args.client_num), m, replace=False)
        #this part should be parallel
        for idx in idxs_clients:
            ClientUpdate(args, agents[idx], idx, net_glob, local_envs[idx], round_, scheduler)
        # weighted = list(map(lambda x: x/sum(weighted), weighted))
        net_glob = ServerUpdate(agents, weighted)
        fresh_agent.policy_net.load_state_dict(net_glob)
        mean_r = eval(args.episode_length, fresh_agent, glob_env, args.eval_episode, name=None, log=False)
        print(f"{fresh_agent.name}, Round:{round_}/{args.round}, Global Env Reward: {mean_r:.2f}")
        ex.log_scalar('Round_test_ep_reward', mean_r, round_)
        # if mean_r > 450:
        #     break

    torch.save(net_glob, model_path + args.filename)
    print("Begin Test:")
    # fresh_agent = DQN(env.observation_space.shape[0], env.action_space.n, args.epsilon, args.local_bc, args.capacity, args.gamma, args.lr, args.device)
    fresh_agent.load(model_path + args.filename)


    eval_agents(args, local_envs, agents, 20, fresh_agent, glob_env) #eval server 20 times
