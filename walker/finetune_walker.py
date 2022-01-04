# -*- coding: utf-8 -*-
# @Time    : 2021/12/10 21:02
# @Author  : Weiming Mai
# @FileName: finetune_walker.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 2021/12/7 15:14
# @Author  : Weiming Mai
# @FileName: td3_walker.py
# @Software: PyCharm


import os
import sys

import numpy as np
sys.path.append(os.path.dirname(sys.path[0]))
from non_stationary_envs.walker import BipedalWalker
from non_stationary_envs.Pendulum import PendulumEnv
from agents.TD3 import TD3
from utils.Tools import try_gpu, set_seed

class Arguments():
    def __init__(self):
        self.local_bc = 256  # local update memory batch size
        self.gamma = 0.98
        self.lr = 0.002
        self.action_bound = 1
        self.tau = 0.01
        self.policy_noise = 0.1 #std of the noise, when update critics
        self.std_noise = 0.25    #std of the noise, when explore
        self.noise_clip = 0.5
        # self.episode_length = 1600 # env._max_episode_steps
        self.eval_episode = 2
        self.episode_length = 1600  # env._max_episode_steps
        self.episode_num = 200
        self.device = try_gpu()
        self.capacity = 1e6
        self.env_seed = 1
        # self.capacity = 10000
        self.C_iter = 5
        self.filename = f"eval_{self.env_seed}_"

model_path = '../outputs/model/walker/'
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

    print(f"eval:{r/args.eval_episode:.2f}")
    return r/args.eval_episode

if __name__ == '__main__':
    args = Arguments()
    env = BipedalWalker()
    set_seed(1)
    # env = PendulumEnv()
    env.reset()
    done = False
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = TD3(state_dim, action_dim, args)
    agent.load(model_path + "eval_None_")
    eval_reward = []
    for i_ep in range(args.episode_num):
        state = env.reset()
        ep_reward = 0
        for iter in range(args.episode_length):

            action = agent.choose_action(state)    # action is array
            n_state, reward, done, _ = env.step(action)  # env.step accept array or list
            # if reward == -100:
            #     reward = -1
            agent.memory.add(state, action, reward, n_state, done)
            agent.finetune()
            ep_reward += reward
            if done == True:
                break
            state = n_state

            # print('obs: {}; reward: {}'.format(observation, reward))
        print(f"episode:{i_ep},train:{ep_reward:.2f}", end = ' ')
        eval_reward.append(eval(agent, env, args))
        if i_ep % 10 == 0:
            agent.save(model_path + args.filename)
        #     np.save(f"{model_path}{args.filename}eval", eval_reward)

        # env.close()
