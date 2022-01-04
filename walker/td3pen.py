


import os
import sys

import numpy as np
sys.path.append(os.path.dirname(sys.path[0]))
# from non_stationary_envs.walker import BipedalWalker, BipedalWalkerHardcore
from non_stationary_envs.Pendulum import PendulumEnv, pendulum_env_config2
from agents.TD3 import TD3
from utils.Tools import try_gpu, set_seed

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
        self.eval_episode = 5
        # self.episode_length = 1600 # env._max_episode_steps
        self.episode_length = 200  # env._max_episode_steps
        self.playing_step = int(2e4)
        self.device = try_gpu()
        # self.device = "cpu"
        self.env_seed = None
        self.C_iter = 2
        # self.capacity = 1e6
        self.capacity = 10000
        self.filename = f"eval_{self.env_seed}_"
model_path = '../outputs/model/pendulum/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

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

if __name__ == '__main__':
    from time import *
    # beg = time()
    args = Arguments()
    # env = BipedalWalker()
    # env = BipedalWalkerHardcore()
    set_seed(1)

    env = PendulumEnv()
    # env = pendulum_env_config2(1)
    print(env.l, env.m, env.g)
    env.seed(1)
    state = env.reset()
    done = False
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = TD3(state_dim, action_dim, args)

    eval_reward = []
    time_step = 0
    n = 0
    Explore(agent, env, args)
    for i_ep in range(args.playing_step):
        time_step += 1
        action = agent.choose_action(state)  # action is array
        n_state, reward, done, _ = env.step(action)  # env.step accept array or list
        # ep_reward += reward
        agent.memory.add(state, action, reward, n_state, done)
        state_batch = agent.UpdateQ()

        if done == True or time_step == args.episode_length:
            state = env.reset()
            # print(ep_reward)
            # ep_reward = 0
            time_step = 0
        else:
            state = n_state
        # eval each agent after local update
        if (i_ep + 1) % args.episode_length == 0:
            n += 1
            reward_log = eval(agent, env, args)
            print(f"eval_episode{n}:{reward_log:.2f}")
            eval_reward.append(reward_log)

    agent.save(model_path + args.filename)
    # state = env.reset()
    # for iter in range(args.episode_length):
    #     action = agent.predict(state)  # action is array
    #     n_state, reward, done, _ = env.step(action)  # env.step accept array or list
    #     env.render()
    #     if done == True:
    #         break
    #     state = n_state
    # env.close()
    # end = time()
    # print(end - beg)
