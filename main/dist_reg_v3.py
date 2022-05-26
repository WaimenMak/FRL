# -*- coding: utf-8 -*-
# @Time    : 2022/3/4 14:42
# @Author  : Weiming Mai
# @FileName: dist_reg_v3.py
# @Software: PyCharm

import os
import sys
import torch
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
from numpy.random import multivariate_normal
import torch.optim as optim
from non_stationary_envs.walker import BipedalWalkerHardcore
from non_stationary_envs.lunar import LunarLanderContinuous
from non_stationary_envs.ContinuousCart import cart_env_config
from models.Network import mlp_policy, distill_qnet as mlp_value#, distill_qnet2 as mlp_value2
from non_stationary_envs.Pendulum import PendulumEnv, pendulum_env_config2
from disttd3v3 import fedTD3, Actor
from utils.Tools import try_gpu, set_seed, ExponentialScheduler, _test
from threading import Thread
from torch.multiprocessing import Pipe, Process, set_start_method
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from utils.Memory import DistilDataset
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser(description='PND')
# parser.add_argument('--beta', type=float, default=0, help='parameter of fedprox')
parser.add_argument('--trial', type=int, default=0, help='test times')
parser.add_argument('--critic_partial', type=float, default=0.1, help='the hyperparameter of the local side distillation')
# parser.add_argument('--mu', type=float, default=0, help='parameter of moon')
parser_args = parser.parse_args()

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class Arguments():
    def __init__(self):
        self.eval_ver = True
        self.gamma = 0.98
        self.lr = 0.002
        self.distil_lr = 0.001
        self.critic_lr = self.lr
        self.epochs = 10
        self.niid = False
        self.schedul = False
        self.std = 0  # not noise, the env params
        self.reweight = False
        ####local distil######
        self.critic_dual = True
        self.critic_epc = 20
        self.critic_partial = 0.1
        self.actor_dual = True
        self.actor_partial = 1
        self.actor_epc = 1
        ######################
        # self.scheduler = ExponentialScheduler(0.002, 0.0004)
        self.env_name = "walker"
        # self.env_name = "lunar"
        # self.env_name = "pendulum"
        # self.env_name = "car"
        # self.env_name = "cart"
        if self.env_name == "pendulum":
            self.reweight_tau = 0.3
            self.epochs = 20
            self.critic_partial = parser_args.critic_partial
            self.niid = True
            self.action_bound = 2
            self.local_bc = 128  # local update memory batch size
            self.episode_length = 200  # env._max_episode_steps
            self.playing_step = int(2e4)
            self.capacity = 10000  #10000
            self.std = 1
            self.noisy_input = False
            self.N = int(100)
            self.M = 2
            self.L = int(100)
            self.policy_noise = 0.2  # std of the noise, when update critics
            self.std_noise = 0.1  # std of the noise, when explore 0.1
        elif self.env_name == "walker":
            self.reweight_tau = 1
            self.epochs = 20
            self.critic_epc = 40
            self.niid = False
            self.action_bound = 1
            self.local_bc = 256  # local update memory batch size
            self.episode_length = 1600  # env._max_episode_steps
            self.playing_step = int(1.2e6)
            self.capacity = 1e6
            self.N = int(400)
            self.M = 2
            self.L = int(400)
            self.noisy_input = False
            self.policy_noise = 0.2  # std of the noise, when update critics
            self.std_noise = 0.1  # std of the noise, when explore 0.1
        elif self.env_name == "lunar":
            self.distil_lr = 0.005
            self.epochs = 20
            self.critic_epc = 30
            self.critic_partial = 0.2
            self.reweight = False
            self.reweight_tau = 1
            self.niid = True
            self.gamma = 0.99
            self.action_bound = 1
            self.local_bc = 256  # local update memory batch size
            self.episode_length = 1200  # env._max_episode_steps
            self.playing_step = int(132000)
            self.capacity = 3.2e4
            self.std = 1
            self.N = int(300)
            self.M = 4
            self.L = int(150)
            self.noisy_input = False
            self.policy_noise = 0.2  # std of the noise, when update critics
            self.std_noise = 0.1  # std of the noise, when explore 0.1
        elif self.env_name == "cart":
            # self.critic_lr = 0.0005
            self.distil_lr = 0.001
            self.epochs = 10
            self.critic_epc = 40
            # self.critic_partial = 0.1
            self.critic_partial = parser_args.critic_partial
            self.reweight = False
            self.reweight_tau = 0.9
            self.niid = True
            self.lr = 0.00009
            # self.lr = 0.0005
            self.action_bound = 1
            self.local_bc = 256  # local update memory batch size
            self.episode_length = 300  # env._max_episode_steps
            self.playing_step = int(3e4)  #2e4
            self.capacity = 10000
            self.std = 1
            self.noisy_input = False
            self.N = int(100)
            self.M = 2
            self.L = int(100)
            self.policy_noise = 0.2  # std of the noise, when update critics
            self.std_noise = 0.1  # std of the noise, when explore 0.1
            # self.schedul = True
            # self.scheduler = ExponentialScheduler(self.lr, 0.0001)
        if self.schedul:
            self.scheduler = ExponentialScheduler(self.lr, self.lr/10)
        else:
            self.scheduler = None

        self.tau = 0.01
        self.noise_clip = 0.5
        self.eval_episode = 5
        self.test_episode = 100

        self.device = try_gpu()
        # self.device = "cuda:0"
        self.mu = 0
        self.beta = 0
        self.alpha = 0
        self.dist = True   #server distillation


        self.Round = self.playing_step // self.N + self.playing_step // self.M // self.L

        self.client_num = 5
        self.env_seed = self.client_num
        # self.env_seed = None
        self.filename = f"v3_distilstd{self.std}_noicy{self.noisy_input}_{self.playing_step}_{self.env_name}{self.env_seed}_N{self.N}_M{self.M}_L{self.L}_criticdual{self.critic_dual}{self.critic_partial}_epc{self.critic_epc}_lr{self.lr}_actordual{self.actor_dual}{self.actor_partial}_reweight{self.reweight}{self.reweight_tau}_distepoch{self.epochs}_lrdecay{self.schedul}"  #filename:env_seed, model_name:env_name

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
            # print(f"mean:{env.mean}", end = " ")
            print(f"params:{env.env_param}")
        elif args.env_name == 'walker':
            env = BipedalWalkerHardcore(seed)
            print(f"r:{env.r}", end = " ")
            print(f"stump:{env.stump_type}")
        elif args.env_name == 'lunar':
            env = LunarLanderContinuous(seed, std=args.std)
            # print(f"noise_mean::{env.mean}")
            print(f"params:{env.env_param}")
        elif args.env_name == 'cart':
            env = cart_env_config(env_seed=seed, std=args.std)
            # print(f"mean:{env.mean}", end=" ")
            print(f"params:{env.env_param}", end=" ")
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
            agent, local_env = agent_env_config(args, seed=(i + 1) * parser_args.trial)
            # agent, local_env = agent_env_config(args, seed=4)
        agent.name = 'agent' + str(i)
        agents.append(agent)
        local_envs.append(local_env)

    return agents, local_envs

def eval(agent, envs, args):
    r = 0
    tau = args.reweight_tau # for softmax weight
    env_num = 0
    weighted = []
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

        weighted.append(temp)
        print(f"env{env_num}:{temp:.2f}", end = ' ')
    # print(f"eval:{r/args.eval_episode:.2f}")
    weighted = [weight + np.abs(np.min(weighted)) for weight in weighted]
    weighted = [np.exp(((1 - weight)/sum(weighted))/tau) for weight in weighted]
    weighted = [weight/sum(weighted) for weight in weighted]
    return r/len(envs), weighted


def Explore(agent, env, args):
    state = env.reset()
    if args.noisy_input:
        state = state + np.random.normal(env.mean, 0.01, state.shape[0])
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
    if args.env_name == "walker":
        seed = client_pipe.recv()
        local_env.seed(seed)
        local_env.modify(seed)
    elif args.env_name == "lunar":
        seed = client_pipe.recv()
        local_env.seed(seed)
        local_env.modify(seed, args.std)

    print(f"{agent.name} in {local_env.env_param}")
    round_q = 0
    q_params, mu_params, frac = client_pipe.recv()
    agent.sync(q_params, mu_params)         # initially synchronize from server

    # ep_reward = 0
    # n = 0
    # N = args.local_bc * args.client_num
    time_step = 0
    eval_reward = []
    state = local_env.reset()
    if args.noisy_input:
        state = state + np.random.normal(local_env.mean, 0.01, state.shape[0])
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
        else:
            agent.memory.add(state, action, reward, n_state, done)
        state_batch, action_batch = agent.UpdateQ()
        # agent.UpdateQ(client_pipe)
        if (i_ep+1) % args.N == 0:  #update Q
            with torch.no_grad():
                dist_rep1, dist_rep2 = agent.critic.Q_net.client_rep(state_batch, action_batch)
                agent.to_cpu([agent.critic.Q_net])
                # dist_rep1 = torch.tensor(dist_rep1.cpu())
                dist_rep1 = dist_rep1.cpu().numpy()
                dist_rep2 = dist_rep2.cpu().numpy()

                mean1 = np.mean(dist_rep1, axis=0)
                mean2 = np.mean(dist_rep2, axis=0)

                cov1 = np.cov(dist_rep1.T)
                cov2 = np.cov(dist_rep2.T)

                # dist_rep1 = torch.from_numpy(
                #     multivariate_normal(mean1, cov1, int(N * frac))).to(torch.float32)
                # dist_rep2 = torch.from_numpy(
                #     multivariate_normal(mean2, cov2, int(N * frac))).to(torch.float32)
                dist_rep1 = torch.from_numpy(
                    multivariate_normal(mean1, cov1, args.local_bc)).to(torch.float32)
                dist_rep2 = torch.from_numpy(
                    multivariate_normal(mean2, cov2, args.local_bc)).to(torch.float32)
                dist_rep1[dist_rep1 < 0] = 0
                dist_rep2[dist_rep2 < 0] = 0
                q_label1, q_label2 = agent.critic.Q_net.server_oupt(dist_rep1, dist_rep2)
            ######## distillation ########

            q_params = agent.critic.Q_net.shared_params()         #q_params state dict of the output layer of Q
            client_pipe.send(((dist_rep1, dist_rep2, q_label1, q_label2), q_params, False))
            prev_q, global_q, frac = client_pipe.recv()  # recv agg Q, prev_q: agg q, global_q: distil q
            ####### local distill #######
            if args.critic_dual:
                agent.critic.Q_net.client_update(prev_q)
                agent.glob_q.cpu()
                agent.glob_q.server_update(agent.critic.Q_net, global_q)
                agent.to_gpu([agent.critic.Q_net, agent.glob_q])

                # partial = 0.1  # default 0.9
                for epc in range(args.critic_epc):
                    agent.critic_distill(args.critic_partial, args.critic_epc)
                # Compare Q value
                if agent.name == "agent0":
                    torch.save(agent.critic.Q_net.state_dict(), f"{model_path}{agent.name}_Q_model_sccd.pth")
            ############################
            else:
                agent.critic.Q_net.client_update(global_q)
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
        if (i_ep+1) % args.episode_length == 0:
            # n += 1
            # reward_log = eval(agent, [local_env], args)
            # print(f"train_episode{n}_{agent.name}:{reward_log:.2f}")
            pass
        # if (i_ep+1) == args.playing_step:
        #     torch.save(dist_rep1, f"{model_path}iidclient{args.client_num}{agent.name}.pt")



def Agg_q(local_models, global_net, weighted, args):
    """
    :param local_models:  tuple local q_net output layer
    :param global_net:   tuple too
    :param weighted: 
    :param args: 
    :return: 
    """
    with torch.no_grad():
        K = args.client_num
        for i in range(2):
            for params in global_net[i].keys():
                global_net[i][params].copy_(weighted[0] * local_models[0][i][params])
            for params in global_net[i].keys():
                for k in range(1, K):
                    global_net[i][params] += weighted[k] * local_models[k][i][params]

def Agg_pi(local_models, global_net, weighted, args):
    with torch.no_grad():
        K = args.client_num
        for params in global_net.keys():
            global_net[params].copy_(weighted[0] * local_models[0][params])
        for params in global_net.keys():
            for k in range(1, K):
                global_net[params] += weighted[k] * local_models[k][params]

def generate_data(statistic, server, local_modelsargs):
    for i in range(args.client_num):
        batch_data1 = torch.from_numpy(np.random.multivariate_normal(statistic[i][0], statistic[i][1], args.local_bc))
        batch_data2 = torch.from_numpy(np.random.multivariate_normal(statistic[i][2], statistic[i][3], args.local_bc))
        q_label1, q_label2 = server.q.server_oupt(batch_data1, batch_data2)
        server.train_dataset.add((batch_data1, batch_data2 , q_label1, q_label2))


class Server():
    def __init__(self,state_dim, action_dim, args):
        self.mu = mlp_policy(state_dim, action_dim, args.action_bound)
        self.q = mlp_value(state_dim, action_dim)
        self.prev_q = deepcopy(self.q)
        for p1, p2 in zip(self.q.feature_q1.parameters(), self.q.feature_q2.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False

        for p1, p2 in zip(self.q.oupt_layer_q1.parameters(), self.q.oupt_layer_q2.parameters()):
            p1.requires_grad = True
            p2.requires_grad = True

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.q.parameters()), lr=args.distil_lr, weight_decay=1e-2)
        # self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.q.parameters()), lr=args.distil_lr)
        self.train_dataset = DistilDataset()
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=args.local_bc, shuffle=True)

    def distill(self, args):

        # train_dataset = TensorDataset(traindata1, traindata2, label1, label2)
        # train_loader = DataLoader(dataset=train_dataset, batch_size=args.local_bc, shuffle=False)
        for epoch in range(args.epochs):
            for data in self.train_loader:
                rep1, rep2, label1, label2 = data
                rep1, rep2, label1, label2 = rep1.to(args.device), rep2.to(args.device), label1.to(args.device), label2.to(args.device)

                oupt1, oupt2 = self.q.server_oupt(rep1, rep2)
                # oupt = torch.cat(oupt)

                self.optimizer.zero_grad()
                loss = self.loss_fn(oupt1, label1) + self.loss_fn(oupt2, label2)
                loss.backward()
                self.optimizer.step()
        # print(f"dist loss:{loss}")

def ServerUpdate(pipe_dict, server, weighted, actor, envs, args): #FedAvg
    """
    A process function
    :param pipe_dict: 
    :param server:  server class
    :param weighted: 
    :param actor:  actor class
    :param env:   local envs
    :param args: 
    :return: 
    """
    eval_freq = args.episode_length // (args.M * args.L)   # transfer round to episode
    eval_reward = []
    local_models = []
    re_weight = weighted
    count = 0
    target = None

    for i in range(args.client_num):
        pipe_dict[i][1].send((server.q.state_dict(), server.mu.state_dict(), re_weight[i]))   #init model

    for round_ in range(args.Round):
        if round_ == int(args.Round * 0.75):
            print("late stage")
        server.train_dataset.clear()
        for i in range(args.client_num):
            distill_data, model, target = pipe_dict[i][1].recv()
            if not target and args.dist:
                server.train_dataset.add(distill_data)
            local_models.append(model)


        if not target:
            Agg_q(local_models, (server.q.oupt_layer_q1.state_dict(), server.q.oupt_layer_q2.state_dict()), re_weight, args)
            # Agg_q(local_models, (server.q.oupt_layer_q1.state_dict(), server.q.oupt_layer_q2.state_dict()), weighted,
            #       args)
            server.prev_q.load_state_dict(server.q.state_dict())
            if args.dist:
                server.q.to(args.device)
                # server.train_dataset.add(distill_data)
                server.distill(args)
                server.q.to("cpu")
            for i in range(args.client_num):
                pipe_dict[i][1].send(((server.prev_q.oupt_layer_q1.state_dict(), server.prev_q.oupt_layer_q2.state_dict()), (server.q.oupt_layer_q1.state_dict(), server.q.oupt_layer_q2.state_dict()), re_weight[i]))  # send q
        else:
            count += 1
            Agg_pi(local_models, server.mu.state_dict(), weighted, args)
            # Agg_pi(local_models, server.mu.state_dict(), re_weight, args)
            for i in range(args.client_num):
                pipe_dict[i][1].send((server.mu.state_dict(), (
                    server.q.oupt_layer_q1.state_dict(), server.q.oupt_layer_q2.state_dict())))  #None -> frac

            actor.policy_net.load_state_dict(server.mu.state_dict())
            # reward_log = eval(actor, envs, args)
            if (count+1) % eval_freq == 0:
                if args.reweight:
                    reward_log, re_weight = eval(actor, envs, args)
                    print(f"{re_weight[3]:.2f}", end=" ")
                else:
                    reward_log, _ = eval(actor, envs, args)
                print(f"mu_round:{count}/{args.playing_step//args.M//args.L} eval_server:{reward_log:.2f}")
                eval_reward.append(reward_log)
                np.save(f"{model_path}{args.filename}_line_{parser_args.trial}", eval_reward)
                # actor.save(f"{model_path}{args.filename}_actor_{parser_args.trial}")
        local_models.clear()
        actor.save(f"{model_path}{args.filename}_actor_{parser_args.trial}")



if __name__ == '__main__':
    # print('adam')
    print(f"gpu:{torch.cuda.device_count()}")
    print(args.device)
    print(f"reweight:{args.reweight_tau}")
    print(model_path + args.filename + f"clients:{args.client_num}")
    print(f"niid:{args.niid}")
    print(f"noise:{args.noisy_input}")
    print(f"dist lr:{args.distil_lr}")
    if args.env_name == "pendulum":
        env = PendulumEnv()
    elif args.env_name == "walker":
        env = BipedalWalkerHardcore()
        print(f"walker version2:{env.v2}")
    elif args.env_name == "cart":
        env = cart_env_config()
    else:
        env = LunarLanderContinuous()
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
                pipe_dict[i][1].send((i + 1) * parser_args.trial)  # seed for box2d class
                # pipe_dict[i][1].send(4)

    glob_thr.start()
    glob_thr.join()
    [p.join() for p in client_process_list]
    print("done!")

    if args.eval_ver:
        # agent = Actor(state_dim, action_dim, args.action_bound, args)
        # agent.load(model_path + f"{model_path}{args.filename}_actor_{parser_args.trial}")
        _test(actor, local_envs, args)

