# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 19:37
# @Author  : Weiming Mai
# @FileName: fedtd3_reg.py
# @Software: PyCharm

import os
import sys
import torch
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
from non_stationary_envs.walker import BipedalWalkerHardcore, walker_config
from non_stationary_envs.lunar import LunarLanderContinuous
from non_stationary_envs.ContinuousCart import cart_env_config
from models.Network import mlp_policy, mlp_value
from non_stationary_envs.Pendulum import PendulumEnv, pendulum_env_config2
from fedregtd3 import fedTD3, Actor
from utils.Tools import try_gpu, set_seed, ExponentialScheduler
# from multiprocessing import Process, Pipe
from threading import Thread
from torch.multiprocessing import Pipe, Process, set_start_method
from copy import deepcopy

try:
    set_start_method('spawn')
except RuntimeError:
    pass

class Arguments():
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.002
        self.niid = False
        self.scheduler = False
        self.std = 0
        # self.scheduler = ExponentialScheduler(0.002, 0.0004)
        # self.env_name = "walker"
        self.env_name = "lunar"
        # self.env_name = "pendulum"
        # self.env_name = "car"
        # self.env_name = "cart"
        if self.env_name == "pendulum":
            self.niid = True
            self.action_bound = 2
            self.local_bc = 128  # local update memory batch size
            self.episode_length = 200  # env._max_episode_steps
            self.playing_step = int(3.2e4)
            self.capacity = 10000
            self.std = 0.1
            self.noisy_input = False
            self.N = int(20)
            self.M = 2
            self.L = int(20)
            self.policy_noise = 0.2  # std of the noise, when update critics
            self.std_noise = 0.1  # std of the noise, when explore 0.1
        elif self.env_name == "walker":
            self.niid = True
            self.action_bound = 1
            self.local_bc = 256  # local update memory batch size
            self.episode_length = 1600 # env._max_episode_steps
            self.playing_step = int(1.2e6)
            self.capacity = 1e6
            self.N = int(400)
            self.M = 2
            self.L = int(400)
            self.noisy_input = False
            self.policy_noise = 0.2  # std of the noise, when update critics
            self.std_noise = 0.1  # std of the noise, when explore 0.1
        elif self.env_name == "lunar":
            self.niid = True
            self.gamma = 0.99
            self.action_bound = 1
            self.local_bc = 256  # local update memory batch size
            self.episode_length = 1200 # env._max_episode_steps
            self.playing_step = int(3.6e5)
            self.capacity = 3.2e4
            self.std = 0
            self.N = int(300)
            self.M = 4
            self.L = int(150)
            self.noisy_input = False
            self.policy_noise = 0.2  # std of the noise, when update critics
            self.std_noise = 0.1  # std of the noise, when explore 0.1
        elif self.env_name == "cart":
            self.niid = True
            self.lr = 0.00009
            self.action_bound = 1
            self.local_bc = 256  # local update memory batch size
            self.episode_length = 200  # env._max_episode_steps
            self.playing_step = int(2e4)
            self.capacity = 10000
            self.std = 0
            self.noisy_input = True
            self.N = int(20)
            self.M = 2
            self.L = int(20)
            self.policy_noise = 0.2  # std of the noise, when update critics
            self.std_noise = 0.1  # std of the noise, when explore 0.1
            # self.scheduler = False
            # self.scheduler = ExponentialScheduler(self.lr, 0.0001)


        self.tau = 0.01
        self.noise_clip = 0.5
        self.eval_episode = 5


        self.device = try_gpu()
        # self.device = "cpu"
        self.mu = 0
        self.beta = 0
        # self.alpha = 0
        self.Round = self.playing_step // self.N + self.playing_step // self.M // self.L

        self.client_num = 1
        self.env_seed = self.client_num
        # self.env_seed = None
        self.filename = f"niidevalfedstd{self.std}_noicy{self.noisy_input}_{self.playing_step}_{self.env_name}{self.env_seed}_N{self.N}_M{self.M}_L{self.L}_beta{self.beta}_mu{self.mu}"  #filename:env_seed, model_name:env_name

args = Arguments()
if args.env_name == "pendulum":
    model_path = '../outputs/fed_model/pendulum/'
elif args.env_name == "walker":
    model_path = '../outputs/fed_model/walker/'
elif args.env_name == "lunar":
    model_path = '../outputs/fed_model/lunar/'
elif args.env_name == "cart":
    model_path = '../outputs/fed_model/cartpole/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

#environment setting
def agent_env_config(args, seed=None):
    env = None
    if seed == None:
        if args.env_name == 'walker':
            env = BipedalWalkerHardcore()
            print(f"r:{env.r}", end=" ")
        elif args.env_name == 'pendulum':
            # env = PendulumEnv()
            env = pendulum_env_config2(seed, std=0)  # seed
        elif args.env_name == 'lunar':
            env = LunarLanderContinuous()
        elif args.env_name == 'cart':
            env = cart_env_config()
    else:
        if args.env_name == 'pendulum':
            env = pendulum_env_config2(seed, std=args.std) # seed
            print(f"mean:{env.mean}", end = " ")
        elif args.env_name == 'walker':
            env = BipedalWalkerHardcore(seed)
            print(f"r:{env.r}", end = " ")
        elif args.env_name == 'lunar':
            env = LunarLanderContinuous(seed, std=args.std)
            # print(f"noise_mean::{env.mean}")
            print(f"params:{env.env_param}")
        elif args.env_name == 'cart':
            env = cart_env_config(env_seed=seed, std=args.std)
            print(f"mean:{env.mean}", end=" ")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # action_dim = env.action_space.n  # 15
    agent = fedTD3(state_dim, action_dim, args)
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
            # agent, local_env = agent_env_config(args, seed=4)
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

            temp += ep_reward/args.eval_episode
            r += ep_reward/args.eval_episode
        print(f"env{env_num}:{temp:.2f}", end = ' ')
    # print(f"eval:{r/args.eval_episode:.2f}")
    return r/len(envs)

def Explore(agent, env, args):
    state = env.reset()
    if args.noisy_input:
        state = state +  np.random.normal(env.mean, 0.01, state.shape[0])
    time_step = 0
    while (len(agent.memory) < args.local_bc):
        time_step += 1
        action = agent.choose_action(state)  # action is array
        n_state, reward, done, _ = env.step(action)  # env.step accept array or list
        if args.noisy_input:
            n_state = n_state + np.random.normal(env.mean, 0.01, state.shape[0])

        if args.env_name == "walker":
            if reward == -100:
                clip_reward = -1
            else:
                clip_reward = reward
            agent.memory.add(state, action, clip_reward, n_state, done)

        elif args.env_name == "car":
            reward_shape = 0
            if n_state[0] > -0.4 and n_state[0] < 0.5:
                reward_shape = 10 * (n_state[0] + 0.4) ** 3
            elif n_state[0] >= env.goal_position:
                reward_shape = 100
            elif n_state[0] <= -0.4:
                reward_shape = -0.1
            agent.memory.add(state, action, reward_shape, n_state, done)
        else:
            agent.memory.add(state, action, reward, n_state, done)
        if done == True or time_step == args.episode_length:
            state = env.reset()
            if args.noisy_input:
                state = state + np.random.normal(env.mean, 0.01, state.shape[0])
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
    if args.env_name == "walker" or args.env_name == "lunar":
        seed = client_pipe.recv()
        local_env.seed(seed)
        local_env.modify(seed)

    print(f"{agent.name} in {local_env.env_param}")
    round_q = 0
    q_params, mu_params = client_pipe.recv()
    agent.sync(q_params, mu_params)

    # ep_reward = 0
    n = 0
    time_step = 0
    # eval_reward = []
    state = local_env.reset()
    if args.noisy_input:
        state = state +  np.random.normal(local_env.mean, 0.01, state.shape[0])
    Explore(agent, local_env, args)
    for i_ep in range(args.playing_step):
        time_step += 1
        action = agent.choose_action(state)    # action is array
        n_state, reward, done, _ = local_env.step(action)  # env.step accept array or list
        if args.noisy_input:
            n_state = n_state + np.random.normal(local_env.mean, 0.01, state.shape[0])
        # ep_reward += reward
        if args.env_name == "walker":
            if reward == -100:
                clip_reward = -1
            else:
                clip_reward = reward
            agent.memory.add(state, action, clip_reward, n_state, done)
        # elif args.env_name == "car":
        #     reward_shape = 0
        #     if n_state[0] > -0.4 and n_state[0] < 0.5:
        #         reward_shape = 10 * (n_state[0] + 0.4) ** 3
        #     elif n_state[0] >= local_env.goal_position:
        #         reward_shape = 100
        #     elif n_state[0] <= -0.4:
        #         reward_shape = -0.1
        #     agent.memory.add(state, action, reward_shape, n_state, done)
        else:
            agent.memory.add(state, action, reward, n_state, done)
        state_batch = agent.UpdateQ()
        # agent.UpdateQ(client_pipe)
        if (i_ep+1) % args.N == 0:  #update Q

            # q = agent.critic.Q_net
            agent.to_cpu([agent.critic.Q_net])
            client_pipe.send((agent.critic.Q_net.state_dict(), False))  # send Q, target: false
            global_q = client_pipe.recv()  # recv agg Q
            agent.critic.Q_net.load_state_dict(global_q)
            # agent.glob_q = global_q
            agent.glob_q.load_state_dict(global_q)
            # for param in agent.critic.Q_net.state_dict().keys():
            #     agent.critic.Q_net.state_dict()[param].copy_(global_q[param])
            agent.to_gpu([agent.critic.Q_net])
            if args.scheduler:
                agent.critic.critic_optimizer.param_groups[0]['lr'] = args.scheduler(round_q)
                round_q += 1
        if (i_ep+1) % args.M == 0:  #update mu and target, target: true
            # agent.DelayUpdate(state_batch, agent.critic.Q_net, agent.tau, client_pipe)
            agent.localDelayUpdate(state_batch, agent.critic.Q_net, agent.tau, client_pipe)

        if done == True or time_step == args.episode_length:
            state = local_env.reset()
            if args.noisy_input:
                state = state + np.random.normal(local_env.mean, 0.01, state.shape[0])
            # print(ep_reward)
            # ep_reward = 0
            time_step = 0
        else:
            state = n_state
        # eval each agent after local update
        if (i_ep+1) % args.episode_length == 0:       # this evaluation maybe necessary if we use local distillation
            n += 1
            # reward_log = eval(agent, [local_env], args)
            # print(f"train_episode{n}_{agent.name}:{reward_log:.2f}")
            # eval_reward.append(reward_log)
            # np.save(f"{model_path}{args.filename}{agent.name}_clientnum{args.client_num}", eval_reward)
            # pass

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

def ServerUpdate(pipe_dict, server, weighted, actor, envs, args): #FedAvg
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
    eval_freq = args.episode_length // (args.M * args.L)   # transfer round to episode
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

            actor.policy_net.load_state_dict(server.mu.state_dict())
            # reward_log = eval(actor, envs, args)
            if (count+1) % eval_freq == 0:
                reward_log = eval(actor, envs, args)
                print(f"mu_round:{count}/{args.playing_step//args.M//args.L} eval_server:{reward_log:.2f}")
                eval_reward.append(reward_log)
                np.save(f"{model_path}{args.filename}server_clientnum{args.client_num}", eval_reward)
                actor.save(f"{model_path}{args.filename}_clientnum{args.client_num}")
        local_models.clear()
        actor.save(f"{model_path}{args.filename}_clientnum{args.client_num}")


if __name__ == '__main__':
    # args = Arguments()
    print(f"niid:{args.niid}")
    print(model_path + args.filename + f"clients:{args.client_num}")
    # env = PendulumEnv()
    # env = mountaincar_config(None)
    # env = BipedalWalkerHardcore()
    env = LunarLanderContinuous()
    # env = cart_env_config()
    set_seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    server = Server(state_dim, action_dim, args)
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

    # server = Server(state_dim, action_dim, args)
    actor = Actor(state_dim, action_dim, args)
    glob_thr = Thread(target=ServerUpdate, args=(pipe_dict, server, weighted, actor, local_envs, args))
    # glob_thr.start()
    [p.start() for p in client_process_list]
    if args.env_name == "walker" or args.env_name == "lunar":
        for i in range(args.client_num):
            if not args.niid:
                pipe_dict[i][1].send(None)  # seed for box2d class
            else:
                pipe_dict[i][1].send(i + 1)  # seed for box2d class
                # pipe_dict[i][1].send(4)

    glob_thr.start()
    glob_thr.join()
    [p.join() for p in client_process_list]
    print("done!")

