# -*- coding: utf-8 -*-
# @Time    : 2021/12/28 16:02
# @Author  : Weiming Mai
# @FileName: central_box2d.py
# @Software: PyCharm


import os
import sys
# import torch
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
# import pickle
from non_stationary_envs.walker import BipedalWalker, BipedalWalkerHardcore
from non_stationary_envs.lunar import LunarLanderContinuous
# from models.Network import mlp_policy, mlp_value
# from non_stationary_envs.Pendulum import PendulumEnv, pendulum_env_config2
# from fedregtd3 import fedTD3 as TD3, Actor
from center_td3 import TD3, Actor
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
        self.niid = False
        self.local_bc = 256  # local update memory batch size
        self.gamma = 0.98
        self.lr = 0.002
        self.env_name = "walker"
        # self.env_name = "lunar"
        if self.env_name == "walker":
            self.niid = True
            self.action_bound = 1
            self.local_bc = 256  # local update memory batch size
            self.episode_length = 1600 # env._max_episode_steps
            self.playing_step = int(1.2e6)
            self.capacity = 1e6
            self.M = 2
            self.noisy_input = False
            self.policy_noise = 0.2  # std of the noise, when update critics
            self.std_noise = 0.1  # std of the noise, when explore 0.1
        elif self.env_name == "lunar":
            self.niid = True
            self.gamma = 0.99
            self.action_bound = 1
            self.local_bc = 256  # local update memory batch size
            self.episode_length = 1200  # env._max_episode_steps
            self.playing_step = int(3.6e5)
            self.capacity = 3.2e4
            self.std = 0
            self.M = 4
            self.noisy_input = False
            self.policy_noise = 0.2  # std of the noise, when update critics
            self.std_noise = 0.1  # std of the noise, when explore 0.1

        # self.action_bound = 1
        # self.action_bound = 2
        # self.beta = 0
        # self.mu = 0
        # self.alpha = 0
        self.tau = 0.01
        # self.policy_noise = 0.2 #std of the noise, when update critics
        # self.std_noise = 0.1    #std of the noise, when explore
        self.noise_clip = 0.5
        self.eval_episode = 5
        # self.episode_length = 1600 # env._max_episode_steps
        # self.episode_length = 200  # env._max_episode_steps
        # self.playing_step = int(2e4)
        # self.playing_step = int(1.2e6)
        self.device = try_gpu()
        # self.device = "cpu"
        self.N = int(0)
        self.L = int(0)
        self.C_iter = self.M
        # self.Round = self.playing_step // self.N + self.playing_step // self.M // self.L
        self.client_num = 5
        self.env_seed = self.client_num
        # self.env_seed = None
        self.filename = f"centerniidstd_noisy{self.noisy_input}_{self.playing_step}_{self.env_name}{self.env_seed}_M{self.M}"


args = Arguments()
model_path = '../outputs/center_model/'
if args.env_name == "walker":
    model_path = '../outputs/center_model/walker/'
elif args.env_name == "lunar":
    model_path = '../outputs/center_model/lunar/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

#environment setting
def agent_env_config(args, seed=None):
    env = None
    if seed == None:
        if args.env_name == 'walker':
            env = BipedalWalkerHardcore()
            print(f"r:{env.r}")
        elif args.env_name == 'lunar':
            env = LunarLanderContinuous()
    else:
        if args.env_name == 'walker':
            env = BipedalWalkerHardcore(seed)
            print(f"r:{env.r},top:{env.stairfreqtop}")
        elif args.env_name == 'lunar':
            env = LunarLanderContinuous(seed, std = args.std)
            print(f"env params:{env.env_param}")
            # print(f"noise_mean::{env.mean}")
    # env.seed(args.train_seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # action_dim = env.action_space.n  # 15
    agent = TD3(state_dim, action_dim, args)
    env.seed(seed)
    return agent, env

def GenerateAgent(args):
    '''
    :param args: 
    :return: local agents and local envs
    '''
    agents = []
    local_envs = []
    for i in range(args.client_num):
        if not args.niid:
            agent, local_env = agent_env_config(args)
        else:
            agent, local_env = agent_env_config(args, seed=i+1)
        agent.name = 'agent' + str(i)
        agents.append(agent)
        local_envs.append(local_env)

    return agents, local_envs

def eval(agent, envs, args):
    r = 0
    env_num = 0
    for env in envs:
        env_num += 1
        temp = 0
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

            temp += ep_reward/args.eval_episode
            r += ep_reward/args.eval_episode
        print(f"env{env_num}:{temp:.2f}", end = ' ')
    # print(f"eval:{r/args.eval_episode:.2f}")
    return r/len(envs)

def Explore(agent, env, args):
    state = env.reset()
    time_step = 0
    while (len(agent.memory) < args.local_bc):
        time_step += 1
        action = agent.choose_action(state)  # action is array
        n_state, reward, done, _ = env.step(action)  # env.step accept array or list
        if reward == -100:
            clip_reward = -1
        else:
            clip_reward = reward
        agent.memory.add(state, action, clip_reward, n_state, done)

        if done == True or time_step == args.episode_length:
            state = env.reset()
            time_step = 0
        else:
            state = n_state

def ClientUpdate(client_pipe, local_env, args):
    """
    A process function
    :param client_pipe: 
    :param args: 
    :param agent: 
    :param local_env: 
    :return: 
    """
    if args.env_name == "walker" or args.env_name == "lunar":
        seed = client_pipe.recv()
        local_env.seed(seed)
        local_env.modify(seed)

    agent = TD3(local_env.observation_space.shape[0], local_env.action_space.shape[0], args)
    print(f"agent in {local_env.env_param}")
    mu_params = client_pipe.recv()
    # agent.sync(q_params, mu_params)
    agent.actor.policy_net.cpu()
    agent.actor.policy_net.load_state_dict(mu_params)
    # agent.critic.Q_net.load_state_dict(q_params)
    # ep_reward = 0
    # n = 0
    time_step = 0
    eval_reward = []
    state = local_env.reset()
    # Explore(agent, local_env, args)
    # n = 0

    for i_ep in range(args.playing_step):
        time_step += 1

        action = agent.choose_action(state)    # action is array
        n_state, reward, done, _ = local_env.step(action)  # env.step accept array or list
        # ep_reward += reward
        if args.env_name == "walker":
            if reward == -100:
                clip_reward = -1
            else:
                clip_reward = reward
            # agent.memory.add(state, action, clip_reward, n_state, done)
            client_pipe.send((state, action, clip_reward, n_state, done))
            # pass
        else:
            client_pipe.send((state, action, reward, n_state, done))

    # if client_pipe.poll():
        pi = client_pipe.recv()
        pi.cpu()
        # agent.actor.policy_net.cpu()
        # actor.policy_net.load_state_dict(pi)
        for param in pi.state_dict().keys():
            agent.actor.policy_net.state_dict()[param].copy_(pi.state_dict()[param])
        # agent.actor.policy_net.cuda()
        # agent.UpdateQ()
        # agent.UpdateQ(client_pipe)
        # agent.localDelayUpdate(state_batch, agent.critic.Q_net, agent.tau, client_pipe)
        if done == True or time_step == args.episode_length:
            state = local_env.reset()

            time_step = 0
        else:
            state = n_state
        # eval each agent after local update
        # if (i_ep+1) % args.episode_length == 0:
        #     n += 1
        #     reward_log = eval(agent, [local_env], args)
        #     print(f"eval_episode{n}_{agent.name}:{reward_log:.2f}")
        #     eval_reward.append(reward_log)
        #     np.save(f"{model_path}{args.filename}{agent.name}_clientnum{args.client_num}", eval_reward)


def ServerUpdate(pipe_dict, server, envs, args): #FedAvg
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
    # eval_reward = []
    # local_models = []
    # count = 0
    # target = None
    # cc=0
    # server.critic.Q_net.cpu()
    server.actor.policy_net.cpu()
    for i in range(args.client_num):
        pipe_dict[i][1].send(server.actor.policy_net.state_dict())   #init model

    # time_step = 1
    eval_reward = []
    n = 0
    for i_ep in range(args.playing_step):
        # server.critic.Q_net.cpu()
        server.actor.policy_net.cpu()

        for i in range(args.client_num):
            # print(pipe_dict[i][1].poll())
        # if pipe_dict[i][1].poll():
            local_data = pipe_dict[i][1].recv()
            server.memory.add(*local_data)
            # pipe_dict[i][1].send(server.actor.policy_net)

        # server.critic.Q_net.to(args.device)
        server.actor.policy_net.to(args.device)
        server.UpdateQ()
        # if len(server.memory) >= args.local_bc:
        #     time_step += 1
        server.actor.policy_net.cpu()
        for i in range(args.client_num):
            pipe_dict[i][1].send(server.actor.policy_net)
            # print(i_ep)
        # agent.UpdateQ(client_pipe)


        if (i_ep+1) % args.episode_length == 0:
            n += 1
            reward_log = eval(server, envs, args)
            print(f"eval_episode{n}_server:{reward_log:.2f}")
            eval_reward.append(reward_log)
            np.save(f"{model_path}{args.filename}server_clientnpum{args.client_num}", eval_reward)
            server.save(f"{model_path}{args.filename}_clientnum{args.client_num}")
    # pickle.dump(server.memory.buffer, open(model_path + 'memory', 'wb'))


if __name__ == '__main__':
    # args = Arguments()
    print(f"niid:{args.niid}")
    print(f"args lr:{args.lr},bc:{args.local_bc}, capacitu:{args.capacity}")
    print(model_path + args.filename + f"clients:{args.client_num}")
    # env = LunarLanderContinuous()
    env = BipedalWalkerHardcore()
    set_seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    server = TD3(state_dim, action_dim, args)
    agents, local_envs = GenerateAgent(args)
    # print('obs: {}; reward: {}'.format(observation, reward))
    # agent.save(model_path + args.filename)
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(args.client_num) for pipe1, pipe2 in (Pipe(),))
    client_process_list = []
    for i in range(args.client_num):
        pro = Process(target=ClientUpdate, args=(pipe_dict[i][0], local_envs[i], args))
        client_process_list.append(pro)

    glob_thr = Thread(target=ServerUpdate, args=(pipe_dict, server, local_envs, args))

    [p.start() for p in client_process_list]
    if args.env_name == "walker" or args.env_name == "lunar":
        for i in range(args.client_num):
            if not args.niid:
                pipe_dict[i][1].send(None)  # seed for box2d class
            else:
                pipe_dict[i][1].send(i + 1)  # seed for box2d class

    glob_thr.start()
    # ServerUpdate(pipe_dict, server, local_envs, args)

    [p.join() for p in client_process_list]
    glob_thr.join()
    print("done!")

