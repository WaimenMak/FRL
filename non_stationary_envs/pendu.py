# -*- coding: utf-8 -*-
# @Time    : 2021/12/27 19:33
# @Author  : Weiming Mai
# @FileName: pendu.py
# @Software: PyCharm



# from fed_reg.center_td3 import Actor
# from fed_reg.fedregtd3 import Actor
from agents.TD3 import Actor
# from agents.fedTD3 import Actor
from utils.Tools import try_gpu
from non_stationary_envs.Pendulum import PendulumEnv, pendulum_env_config2
from non_stationary_envs.walker import BipedalWalkerHardcore, BipedalWalker
from non_stationary_envs.lunar import LunarLanderContinuous
from non_stationary_envs.ContinuousCart import cart_env_config
import pandas as pd
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
        self.env_name = "walker"
        # self.env_name = "lunar"
        # self.env_name = "pendulum"
        # self.env_name = "cartpole"
        if self.env_name == "pendulum":
            self.action_bound = 2
            self.local_bc = 64  # local update memory batch size
            self.episode_length = 200  # env._max_episode_steps
            self.playing_step = int(2e4)
            self.capacity = 10
            self.std = 0
            self.noisy_input = True
            self.N = int(0)
            self.M = 2
            self.L = int(0)
        elif self.env_name == "walker":
            self.action_bound = 1
            self.local_bc = 256  # local update memory batch size
            self.episode_length = 1600 # env._max_episode_steps
            self.playing_step = int(1e6)
            self.capacity = 1
            self.N = int(0)
            self.M = 2
            self.L = int(0)
            self.noisy_input = False
        elif self.env_name == "lunar":
            self.action_bound = 1
            self.local_bc = 256  # local update memory batch size
            self.episode_length = 1200  # env._max_episode_steps
            self.playing_step = int(3.6e5)
            self.capacity = 1
            self.N = int(0)
            self.L = int(0)
            self.M = 4
            self.std = 0
            self.noisy_input = False
        elif self.env_name == "cartpole":
            self.action_bound = 1
            self.local_bc = 0  # local update memory batch size
            self.episode_length = 200  # env._max_episode_steps
            self.playing_step = int(2e4)
            self.capacity = 10
            self.std = 1
            self.noisy_input = False
            self.N = int(0)
            self.M = 2
            self.L = int(0)


        self.device = try_gpu()
        self.beta = 0
        self.C_iter = self.M
        self.env_seed = None
        self.beta = 0
        self.mu = 0
        self.alpha = 0
        self.client_num = 5
        # self.capacity = 10000
        # self.episode_length = 200  # env._max_episode_steps
        self.eval_episode = 100
        self.filename = []

        #cart
        if self.env_name == "cartpole" and self.noisy_input == False:
            # self.filename.append("niidevalfedstd1_noicyFalse_20000_cart5_N20_M2_L20_beta0_mu0_dual_False_clientnum5actor_") #fedavg
            # self.filename.append("niidevalfedstd1_noicyFalse_20000_cart5_N20_M2_L20_beta0_mu0.01_dual_False_clientnum5actor_")  # moon
            # self.filename.append("niidevalfedstd1_noicyFalse_20000_cart5_N20_M2_L20_beta0.01_mu0_dual_False_clientnum5actor_")  # fedprox
            # self.filename.append("cartstd1niidevalfedscaffold_cart5_N20_M2_L20_clientnum5actor_")   #scaffold
            # self.filename.append("distilstd1_noicyFalse_20000_cart5_N20_M2_L20_dualFalse_reweight0.5_distepoch10_clientnum5actor_") #dist
            # self.filename.append("distilstd1_noicyFalse_20000_cart5_N20_M2_L20_dualTrue_reweightTrue_distepoch10_clientnum5actor_") #dist dual
            # self.filename.append(
            #     "v2_distilstd1_noicyFalse_20000_cart5_N20_M2_L20_dualFalse_reweight0.5_distepoch10_clientnum5actor_")  # dist stat
            # self.filename.append("centerniidstd_noisyFalse_20000_cart5_M2_clientnum5actor_")  #center iid

            self.filename.append("reweight0.5niidfedstd1_noicyFalse_20000_cart3_N20_M2_L20_beta0_mu0_dual_False_lrdecayFalse_clientnum3actor_")
            # self.filename.append("reweight0.5niidfedstd1_noicyFalse_20000_cart3_N20_M2_L20_beta0_mu0_dual_False_clientnum3actor_")

        #cart noise
        elif self.env_name == "cartpole" and self.noisy_input == True:
            self.filename.append("niidevalfedstd0_noicyTrue_20000_cart5_N20_M2_L20_beta0_mu0_dist_False_clientnum5actor_") #fedavg
            self.filename.append("niidevalfedstd0_noicyTrue_20000_cart5_N20_M2_L20_beta0_mu0.01_dual_False_clientnum5actor_")  # moon
            self.filename.append("niidevalfedstd0_noicyTrue_20000_cart5_N20_M2_L20_beta0.01_mu0_dual_False_clientnum5actor_")  # fedprox
            self.filename.append("fedscaffold_std0_noiseTrue_cart5_N20_M2_L20_clientnum5actor_")   #scaffold
            self.filename.append("v2_distilstd1_noicyFalse_30000_pendulum5_N100_M2_L100_dualFalse_reweight0.3_distepoch20_clientnum5actor_") #dist
            # self.filename.append("distilstd0_noicyTrue_20000_cart5_N20_M2_L20_dualTrue_reweightTrue_distepoch10_clientnum5actor_") #dist dual
            self.filename.append(
                "v2_distilstd1_noicyFalse_30000_pendulum5_N100_M2_L100_dualFalse_reweight0.3_distepoch20_clientnum5actor_")  # dist stat
            self.filename.append("centerniidstd_noisyTrue_20000_cart5_M2_clientnum5actor_")  #center iid

        #pendulum
        elif self.env_name == "pendulum" and self.noisy_input == False:
            self.filename.append("niidevalfedstd1_noicyFalse_30000_pendulum5_N100_M2_L100_beta0_mu0_dual_False_clientnum5actor_") #fedavg
            self.filename.append("niidevalfedstd1_noicyFalse_30000_pendulum5_N100_M2_L100_beta0_mu0.01_dual_False_clientnum5actor_")  # moon
            self.filename.append("niidevalfedstd1_noicyFalse_30000_pendulum5_N100_M2_L100_beta0.01_mu0_dual_False_clientnum5actor_")  # fedprox
            self.filename.append("fedscaffold_std1_noiseFalse_pendulum5_N100_M2_L100_clientnum5actor_")   #scaffold
            self.filename.append("distilstd1_noicyFalse_30000_pendulum5_N100_M2_L100_dualFalse_reweight0.3_distepoch20_clientnum5actor_") #dist

            self.filename.append(
                "v2_distilstd1_noicyFalse_30000_pendulum5_N100_M2_L100_dualFalse_reweight0.3_distepoch20_clientnum5actor_")  # dist stat
            self.filename.append("centerniidstd1_noisyFalse_32000_pendulum5_M2_clientnum5actor_")  #center iid
        elif self.env_name == "pendulum" and self.noisy_input == True:
            self.filename.append(
                "niidevalfedstd0_noicyTrue_30000_pendulum5_N100_M2_L100_beta0_mu0_dual_False_clientnum5actor_")
            self.filename.append(
                "v2_distilstd0_noicyTrue_30000_pendulum5_N100_M2_L100_dualFalse_reweight0.3_distepoch20_clientnum5actor_")


        elif self.env_name == "walker":
            # self.filename.append("v3niidevalfedstd0_noicyFalse_1200000_walker5_N400_M2_L400_beta0_mu0_dual_False_clientnum5actor_") #fedavg v3
            # self.filename.append("v3_distilstd0_noicyFalse_1200000_walker5_N400_M2_L400_criticdualTrue0.1_actordualFalse0.9_reweightFalse1_distepoch20_lrdecayFalse_actor_1actor_")
            # self.filename.append("v2_distilstd0_noicyFalse_1200000_walker5_N400_M2_L400_dualFalse_reweight0.5_distepoch20_clientnum5actor_")
            # self.filename.append(
                # "niidevalfedstd0_noicyFalse_1200000_walker5_N400_M2_L400_beta0_mu0.01_dual_False_clientnum5actor_") #moon
            # self.filename.append(
            #     "v3niidevalfedstd0_noicyFalse_1200000_walker5_N400_M2_L400_beta0_mu0_dual_False_clientnum5actor_") #fedavg
            self.filename.append(
                "v3_distilstd0_noicyFalse_1200000_walker5_N400_M2_L400_criticdualTrue0.1_actordualFalse0.9_reweightFalse1_distepoch20_lrdecayFalse_actor_1actor_") #best walker dist
            # self.filename.append(
            #     "reweight0.5niidfedstd0_noicyFalse_1200000_walker5_N400_M2_L400_beta0_mu0_dual_False_clientnum5actor_") #

args = Arguments()
if args.env_name == "cartpole":
    model_path = '../outputs/fed_model/cartpole/'
elif args.env_name == "pendulum":
# model_path = '../outputs/center_model/pendulum/'
    model_path = '../outputs/fed_model/pendulum/'
elif args.env_name == "walker":
    model_path = '../outputs/fed_model/walker/'
# model_path = '../outputs/fed_model/lunar/'
# model_path = '../outputs/center_model/walker/'
# model_path = '../outputs/center_model/lunar/'
# model_path = '../outputs/model/walker/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

def agent_env_config(args, seed=None):
    env = None
    if seed == None:
        if args.env_name == 'walker':
            env = BipedalWalkerHardcore()
            print(f"r:{env.r}")
        elif args.env_name == 'pendulum':
            # env = PendulumEnv()
            env = pendulum_env_config2(seed, std=0)  # seed
        elif args.env_name == 'lunar':
            env = LunarLanderContinuous()
        elif args.env_name == "cartpole":
            env = cart_env_config()
    else:
        if args.env_name == 'pendulum':
            env = pendulum_env_config2(seed, std=args.std) # seed
            print(f"params:{env.env_param}")
        elif args.env_name == 'walker':
            env = BipedalWalkerHardcore(seed)
            print(f"r:{env.r}")
        elif args.env_name == 'lunar':
            env = LunarLanderContinuous(seed, std=args.std)
            # print(f"noise_mean::{env.mean}")
            print(f"params:{env.env_param}")
        elif args.env_name == "cartpole":
            env = cart_env_config(env_seed=seed, std=args.std)
            print(f"params:{env.env_param}")
    env.seed(seed)
    return env

def GenerateAgent(args):
    '''
    :param args: 
    :return: local agents and local envs
    '''
    # agents = []
    local_envs = []
    for i in range(args.client_num):
        # local_env = agent_env_config(args)
        local_env = agent_env_config(args, seed=i+1)
        agent.name = 'agent' + str(i)
        # agents.append(agent)
        local_envs.append(local_env)

    return local_envs

def eval(agent, envs, args):

    env_num = 0
    std_list = [] #over all std
    mean_list = []
    each_env = []
    total = []
    for env in envs:
        env_num += 1
        r = 0
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
            r += ep_reward / args.eval_episode
            # temp += ep_reward/args.eval_episode
            each_env.append(ep_reward)

            total.append(ep_reward)
        # each_env.append(temp)
        mean = np.mean(each_env)
        std = np.std(each_env)

        std_list.append(mean)
        mean_list.append(f"{mean:.2f}+-{std:.2f}")
        print(f"env{env_num}:mean {mean:.2f}, std {std:.2f}", end=" ")
        each_env.clear()

    # print(f"eval:{r/args.eval_episode:.2f}")

    mean_list.append(f"{np.mean(total):.2f}+-{np.std(std_list):.2f}")
    print(f"overall mean:{np.mean(total):.2f}, std {np.std(std_list):.2f}")
    return mean_list

if __name__ == '__main__':

    np.random.seed(1)
    # env = BipedalWalkerHardcore()
    if args.env_name == "pendulum":
        env = PendulumEnv()
    elif args.env_name == "walker":
        env = BipedalWalkerHardcore()
        print(f"walker version2:{env.v2}")
    elif args.env_name == "cartpole":
        env = cart_env_config()
    else:
        env = LunarLanderContinuous()
    env.reset()
    done = False
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = Actor(state_dim, action_dim, args.action_bound, args)
    # agent = Actor(state_dim, action_dim, args)
    # agent.load(f"{model_path}{args.filename}")
    result = []
    n = 0
    local_envs = GenerateAgent(args)
    for file in args.filename:
        n += 1
        # if n == len(args.filename):
        #     if args.env_name == "cartpole":
        #         model_path = '../outputs/center_model/cartpole/'
        #     elif args.env_name == "pendulum":
        #         model_path = '../outputs/center_model/pendulum/'
        #     elif args.env_name == "walker":
        #         model_path = '../outputs/center_model/walker/'

        agent.load(model_path+file)
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


        mean_list = eval(agent, local_envs, args)
        result.append(mean_list)




    # rd = pd.DataFrame(result, columns=["env1", "env2", "env3", "env4", "env5", "overall"], index = ["fedavg","moon",
    #                                                                                                 "fedprox","scaffold","dist","dist dual", "dist stat", "central"])

    # rd = pd.DataFrame(result, columns=["env1", "env2", "env3", "env4", "env5", "overall"], index=["fedavg", "moon",
    #                                                                                               "fedprox", "scaffold",
    #                                                                                               "dist",
    #                                                                                               "dist stat",
    #                                                                                               "central"])
    # rd.to_excel("result.xlsx", index=True)