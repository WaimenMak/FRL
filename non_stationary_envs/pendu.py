# -*- coding: utf-8 -*-
# @Time    : 2021/12/27 19:33
# @Author  : Weiming Mai
# @FileName: pendu.py
# @Software: PyCharm



# from fed_reg.center_td3 import Actor
from fed_reg.fedregtd3 import Actor
# from agents.fedTD3 import Actor
from utils.Tools import try_gpu
from non_stationary_envs.Pendulum import PendulumEnv, pendulum_env_config2
from non_stationary_envs.walker import BipedalWalkerHardcore
import numpy as np
import os
class Arguments():
    def __init__(self):
        # self.local_bc = 256  # local update memory batch size
        # self.gamma = 0.98
        self.lr = 0.002
        # self.action_bound = 2
        self.tau = 0.01
        self.policy_noise = 0.01 #std of the noise, when update critics
        self.std_noise = 0.01    #std of the noise, when explore
        self.noise_clip = 0.5
        # self.episode_length = 1600 # env._max_episode_steps
        # self.env_name = "walker"
        self.env_name = "pendulum"
        if self.env_name == "pendulum":
            self.action_bound = 2
            self.local_bc = 64  # local update memory batch size
            self.episode_length = 200  # env._max_episode_steps
            self.playing_step = int(2e4)
            self.capacity = 10000
            self.N = int(20)
            self.M = 2
            self.L = int(20)
        elif self.env_name == "walker":
            self.action_bound = 1
            self.local_bc = 256  # local update memory batch size
            self.episode_length = 1600 # env._max_episode_steps
            self.playing_step = int(1e6)
            self.capacity = 1e6
            self.N = int(400)
            self.M = 2
            self.L = int(400)

        self.device = try_gpu()
        self.beta = 0
        self.C_iter = self.M
        self.env_seed = None
        self.beta = 0
        self.mu = 0
        self.client_num = 5
        # self.capacity = 10000
        # self.episode_length = 200  # env._max_episode_steps
        self.eval_episode = 100
        # self.playing_step = int(2e4)
        # self.filename = f"eval_{self.env_seed}_"
        # self.filename = "niidevalfedreg_pendulum5_N10_M2_L20_beta0.5_clientnum5" #fed
        self.filename = "niidevalfed3e4_pendulum5_N10_M2_L20_beta0_mu0_clientnum5" #fed"  # fed
        # self.filename = "centerniid_pendulum5_M2_clientnum5"   #center niid
        # self.filename = "normalevalfedreg_pendulum5_M2_clientnum5"  #center iid
        # self.filename = "evalfedreg_pendulum5_N10_M2_L20_beta0.0_clientnum5" #fedavg
        # self.filename = "niidevalfedmoon_pendulum5_N10_M2_L20_beta0_mu1_clientnum5" #moon
        # self.filename = "niidevalfedmoon_pendulum5_N10_M2_L20_beta0_mu1clientnum5"  # moon
        # self.filename = "niidevalfedmoon_pendulum5_N20_M2_L20_beta0.5_mu0_clientnum5"  # fedprox
        # self.filename = "niidevalfed_walker5_N400_M2_L400_beta0_mu0.01_clientnum5" #fedavg
        # self.filename = "centeriid_walkerNone_M5_clientnum5"
        # self.filename = "centerniid_walkerNone_M2_clientnum5"



# model_path = '../outputs/center_model/walker/'
model_path = '../outputs/fed_model/pendulum/'
# model_path = '../outputs/fed_model/walker/'
# model_path = '../outputs/model/pendulum/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

def agent_env_config(args, seed=None):
    env = None
    if seed == None:
        if args.env_name == 'walker':
            env = BipedalWalkerHardcore()
            print(f"r:{env.r}")
        elif args.env_name == 'pendulum':
            env = PendulumEnv()
    else:
        if args.env_name == 'pendulum':
            env = pendulum_env_config2(seed) # seed
        elif args.env_name == 'walker':
            env = BipedalWalkerHardcore(seed)
            # env = BipedalWalkerHardcore(4)
            print(f"r:{env.r}")
    # env.seed(args.train_seed)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # action_dim = env.action_space.n  # 15
    # agent = fedTD3(state_dim, action_dim, args)
    env.seed(1)
    return env

def GenerateAgent(args):
    '''
    :param args: 
    :return: local agents and local envs
    '''
    # agents = []
    local_envs = []
    for i in range(args.client_num):
        # agent, local_env = agent_env_config(args)
        local_env = agent_env_config(args, seed=i+1)
        # agent.name = 'agent' + str(i)
        # agents.append(agent)
        local_envs.append(local_env)

    return local_envs

def eval(agent, envs, args):

    env_num = 0
    each_env = []
    total = []
    for env in envs:
        env_num += 1
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
            r += ep_reward / args.eval_episode
            # temp += ep_reward/args.eval_episode
            each_env.append(ep_reward)

            total.append(ep_reward)
        # each_env.append(temp)
        print(f"env{env_num}:mean {np.mean(each_env):.2f}, std {np.std(each_env):.2f}", end = ' ')
        each_env.clear()

    # print(f"eval:{r/args.eval_episode:.2f}")
    return total

if __name__ == '__main__':
    args = Arguments()
    # env = BipedalWalkerHardcore()
    env = PendulumEnv()
    # env = pendulum_env_config2(3)
    env.reset()
    done = False
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = Actor(state_dim, action_dim, args)
    # agent = Actor(state_dim, action_dim, args)
    # agent.load(f"{model_path}{args.filename}")
    agent.load(model_path+args.filename)
    # state = env.reset()
    # ep_reward = 0
    # for iter in range(args.episode_length):
    #
    #     env.render()
    #     action = agent.predict(state)  # action is array
    #     n_state, reward, done, _ = env.step(action)  # env.step accept array or list
    #     print(reward)
    #     ep_reward += reward
    #     if done == True:
    #         break
    #     state = n_state
    # print(ep_reward)
    # env.close()

    local_envs = GenerateAgent(args)
    ex = eval(agent, local_envs, args)
    print(f"overall mean:{np.mean(ex):.2f}, std {np.std(ex):.2f}")