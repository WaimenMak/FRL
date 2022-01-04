# -*- coding: utf-8 -*-
# @Time    : 2021/12/21 16:51
# @Author  : Weiming Mai
# @FileName: fed2_td3.py
# @Software: PyCharm


import os
import sys
import torch
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
from non_stationary_envs.walker import BipedalWalker
from models.Network import mlp_policy, mlp_value
from non_stationary_envs.Pendulum import PendulumEnv
# from agents.fedTD3 import fedTD3, Actor
# from agents.fedTD3_cpu import fedTD3, Actor
from agents.fedTD3_gpu import fedTD3, Actor
from utils.Tools import try_gpu, set_seed
# from multiprocessing import Process, Pipe
from threading import Thread
from torch.multiprocessing import Pipe, Process, set_start_method

try:
     set_start_method('spawn')
except RuntimeError:
    pass

class Arguments():
    def __init__(self):
        self.local_bc = 64  # local update memory batch size
        self.gamma = 0.98
        self.lr = 0.002
        self.action_bound = 2
        self.tau = 0.01
        self.policy_noise = 0.2 #std of the noise, when update critics
        self.std_noise = 0.1    #std of the noise, when explore
        self.noise_clip = 0.5
        self.eval_episode = 2
        # self.episode_length = 1600 # env._max_episode_steps
        self.episode_length = 200  # env._max_episode_steps
        self.playing_step = int(2e4)
        self.device = try_gpu()
        # self.device = "cpu"
        self.env_seed = None
        # self.capacity = 1e6
        self.capacity = 10000
        self.N = int(20)   #send q frequency (playing_step // N), playing_step:update q frequency
        self.M = 2   #update mu frequency (playing_step // M)
        self.L = int(20)   #send mu frequency (playing_step // M // L)
        self.Round = self.playing_step // self.N + self.playing_step // self.M // self.L
        # self.C_iter = 5
        self.client_num = 3
        self.filename = f"eval_{self.env_seed}_N{self.N}_M{self.M}_L{self.L}"  #filename:env_seed, model_name:env_name

model_path = '../outputs/fed_model/pendulum/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

#environment setting
def agent_env_config(args, seed):
    env = None
    if seed == None:
        # env = gym.make(env_name)
        env = PendulumEnv()
    else:
        # env = pendulum_env_config(seed) # seed
        pass
    # env.seed(args.train_seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # action_dim = env.action_space.n  # 15
    agent = fedTD3(state_dim, action_dim, args)
    return agent, env

def GenerateAgent(args):
    '''
    :param args: 
    :return: local agents and local envs
    '''
    agents = []
    local_envs = []
    for i in range(args.client_num):
        agent, local_env = agent_env_config(args, seed=args.env_seed)
        agent.name = 'agent' + str(i)
        agents.append(agent)
        local_envs.append(local_env)

    return agents, local_envs

def eval(agent, env, args):
    r = 0
    for i_ep in range(args.eval_episode):
        state = env.reset()
        ep_reward = 0
        for iter in range(args.episode_length):
            action = agent.predict(state)  # action is array
            n_state, reward, done, _ = env.step(action)  # env.step accept array or list
            ep_reward += reward
            if done == True:
                break
            state = n_state

        r += ep_reward
    # print(f"eval:{r/args.eval_episode:.2f}")
    return r/args.eval_episode

def Explore(agent, env, args):
    state = env.reset()
    time_step = 0
    while (len(agent.memory) < args.local_bc):
        time_step += 1
        action = agent.choose_action(state)  # action is array
        n_state, reward, done, _ = env.step(action)  # env.step accept array or list
        agent.memory.add(state, action, reward, n_state, done)
        if done == True or time_step == args.episode_length:
            state = env.reset()
            time_step = 0
        else:
            state = n_state

def ClientUpdate(client_pipe, agent, local_env, args):
    """
    A process function
    :param client_pipe: 
    :param args: 
    :param agent: 
    :param local_env: 
    :return: 
    """
    q_params, mu_params = client_pipe.recv()
    agent.sync(q_params, mu_params)

    # ep_reward = 0
    n = 0
    time_step = 0
    eval_reward = []
    state = local_env.reset()
    Explore(agent, local_env, args)
    for i_ep in range(args.playing_step):
        time_step += 1
        # print(i_ep+1)
        action = agent.choose_action(state)    # action is array
        n_state, reward, done, _ = local_env.step(action)  # env.step accept array or list
        # ep_reward += reward
        agent.memory.add(state, action, reward, n_state, done)
        # state_batch = agent.UpdateQ()
        agent.UpdateQ(args, client_pipe, i_ep+1)
        # agent.UpdateQ(client_pipe)
        if (i_ep+1) % args.N == 0:  #update Q
            # q = agent.critic.Q_net
            agent.to_cpu([agent.critic.Q_net])
            client_pipe.send((agent.critic.Q_net.state_dict(), False))  # send Q, target: false
            global_q = client_pipe.recv()                     # recv agg Q
            # agent.critic.Q_net.load_state_dict(global_q)
            for param in agent.critic.Q_net.state_dict().keys():
                agent.critic.Q_net.state_dict()[param].copy_(global_q[param])
            agent.to_gpu([agent.critic.Q_net])

        # if (i_ep+1) % args.M == 0:  #update mu and target, target: true
        #     # agent.DelayUpdate(state_batch, agent.tau, client_pipe)
        #     agent.localDelayUpdate(state_batch, agent.critic.Q_net, agent.tau, client_pipe)

        if done == True or time_step == args.episode_length:
            state = local_env.reset()
            # print(ep_reward)
            # ep_reward = 0
            time_step = 0
        else:
            state = n_state
        # eval each agent after local update
        if (i_ep+1) % args.episode_length == 0:
            n += 1
            reward_log = eval(agent, local_env, args)
            # print(f"eval_episode{n}_{agent.name}:{reward_log:.2f}")
            eval_reward.append(reward_log)
            np.save(f"{model_path}{args.filename}{agent.name}_clientnum{args.client_num}", eval_reward)

def Agg(local_models, global_net, weighted, args):
    with torch.no_grad():
        K = args.client_num
        for params in global_net.keys():
            global_net[params].copy_(weighted[0] * local_models[0][params])
        for params in global_net.keys():
            for k in range(1, K):
                global_net[params] += weighted[k] * local_models[k][params]

class Server():
    def __init__(self,state_dim, action_dim, args):
        self.mu = mlp_policy(state_dim, action_dim, args.action_bound)
        self.q = mlp_value(state_dim, action_dim)

def ServerUpdate(pipe_dict, server, weighted, actor, env, args): #FedAvg
    """
    A process function
    :param pipe_dict: 
    :param server:  server class
    :param weighted: 
    :param actor:  actor class
    :param env: 
    :param args: 
    :return: 
    """
    eval_reward = []
    local_models = []
    count = 0
    target = None
    for i in range(args.client_num):
        pipe_dict[i][1].send((server.q.state_dict(), server.mu.state_dict()))   #init model

    for round_ in range(args.Round):
        for i in range(args.client_num):
            model, target = pipe_dict[i][1].recv()
            local_models.append(model)

        if not target:
            Agg(local_models, server.q.state_dict(), weighted, args)
            for i in range(args.client_num):
                pipe_dict[i][1].send(server.q.state_dict())  # send q
        else:
            count += 1
            Agg(local_models, server.mu.state_dict(), weighted, args)
            for i in range(args.client_num):
                pipe_dict[i][1].send((server.mu.state_dict(), server.q.state_dict()))  # send q and mu

            # actor.policy_net.load_state_dict(server.mu.state_dict())
            for param in (actor.policy_net.state_dict().keys()):
                actor.policy_net.state_dict()[param].copy_(server.mu.state_dict()[param])
            reward_log = eval(actor, env, args)
            if (count+1) % 10 == 0:
                print(f"mu_round:{count}/{args.playing_step//args.M//args.L} eval_server:{reward_log:.2f}")
            eval_reward.append(reward_log)
            np.save(f"{model_path}{args.filename}server_clientnum{args.client_num}", eval_reward)

        local_models.clear()
    actor.save(f"{model_path}{args.filename}_clientnum{args.client_num}")


if __name__ == '__main__':
    args = Arguments()
    env = PendulumEnv()
    env.seed(1)
    # env = BipedalWalker()
    set_seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agents, local_envs = GenerateAgent(args)
    weighted = [agent.batch_size for agent in agents]
    weighted = list(map(lambda x: x / sum(weighted), weighted))
    # print('obs: {}; reward: {}'.format(observation, reward))
    # agent.save(model_path + args.filename)
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(args.client_num) for pipe1, pipe2 in (Pipe(),))
    client_process_list = []
    for i in range(args.client_num):
        pro = Process(target=ClientUpdate, args=(pipe_dict[i][0], agents[i], local_envs[i], args))
        client_process_list.append(pro)

    server = Server(state_dim, action_dim, args)
    actor = Actor(state_dim, action_dim, args)
    glob_thr = Thread(target=ServerUpdate, args=(pipe_dict, server, weighted, actor, env, args))
    glob_thr.start()
    [p.start() for p in client_process_list]

    glob_thr.join()
    [p.join() for p in client_process_list]
    print("done!")


