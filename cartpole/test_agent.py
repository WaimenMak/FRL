# -*- coding: utf-8 -*-
# @Time    : 2021/11/10 19:56
# @Author  : Weiming Mai
# @FileName: test_agent.py
# @Software: PyCharm

# import pynvml
import os
from non_stationary_envs.Cartpole import CartPoleEnv, cart_env_config
from copy import deepcopy
from sacred import Experiment
from sacred.observers import MongoObserver

from utils.Tools import try_gpu, set_seed, FractionScheduler
# import random
from agents.DQN import fed_DQN
from models.Network import MLP

#save model
model_path = '../outputs/fed_model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


ex = Experiment("test_client10")  #name of the experiment
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
    # args.episode_num = 50
    args.eval_episode = 30
    args.episode_length = 500
    args.capacity = 1
    args.eps_decay = None
    # args.C_iter = 5
    # args.train_seed = 1
    # args.predict_seed = 10
    # args.show_memory = False
    # args.seed = 1
    # args.env_name = 'CartPole-v1'
    # args.frac = 1      # [0,1] choosing agents
    # args.client_num = 1
    args.device = try_gpu()
    seed = 1
    #FedAvg Parameters

def eval(length, agent, env, eval_episode, name, log=True):
    r = 0
    for i_ep in range(eval_episode):
        state = env.reset()
        ep_reward = 0
        for _ in range(length):
            # env.render()
            action = agent.predict(state)
            n_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = n_state
            if done == True:
                break
        if log:
            ex.log_scalar(name, ep_reward)  # just for eval agent in local envs
        r += ep_reward

    return r/eval_episode

def eval_agents(args, local_envs, agents, eval_episode, fresh_agent):
    k = 0
    total_mean_reward = 0
    total_agents = deepcopy(agents)
    total_envs = deepcopy(local_envs)
    total_agents.append(fresh_agent)
    # total_envs.append(glob_env)
    for env in total_envs:
        for agent in total_agents:
            R = eval(args.episode_length, agent, env, eval_episode, agent.name+'_test_env' + str(k)) #'agent0_test_env0'
            ex.log_scalar(agent.name+'_test_env_mean_reward' + str(k), R)
            total_mean_reward += R
            print(f"{agent.name}, Env:{k}/{len(local_envs)}, Reward: {R:.2f}")
        k += 1
    return total_mean_reward/(k+1)
@ex.automain
def main_eval(args):
    # glob_env = gym.make(args.env_name)
    # glob_env = cart_env_config(0)
    glob_env = CartPoleEnv()
    fresh_agent = fed_DQN(glob_env.observation_space.shape[0], glob_env.action_space.n, args)

    fresh_agent.name = 'agent_server'
    # file_name = 'seed_7_feddqn.pth'
    file_name = 'clients_10_fedavgdqn.pth'
    # file_name = 'feddqn3.pth'
    print(file_name)
    fresh_agent.load(model_path + file_name)
    # fresh_agent.load(model_path + 'clients_5_fedavgdqn.pth')
    local_envs = []
    agents = []
    for i in range(10):
        local_envs.append(cart_env_config(i))
    print("Average Reward:{:.2f}".format(eval_agents(args, local_envs, agents, 50, fresh_agent)))