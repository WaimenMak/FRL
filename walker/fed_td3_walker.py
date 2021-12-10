# -*- coding: utf-8 -*-
# @Time    : 2021/12/9 20:20
# @Author  : Weiming Mai
# @FileName: fed_td3_walker.py
# @Software: PyCharm

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from non_stationary_envs.walker import BipedalWalker
from non_stationary_envs.Pendulum import PendulumEnv
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
        # self.episode_length = 1600 # env._max_episode_steps
        self.episode_length = 200  # env._max_episode_steps
        self.playing_step = int(1e4)
        self.device = try_gpu()
        # self.capacity = 1e6
        self.capacity = 10000
        self.C_iter = 5
        self.filename = f"eval_{self.env_seed}_"

model_path = '../outputs/fed_model/walker/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


if __name__ == '__main__':
    args = Arguments()
    # env = BipedalWalker()
    set_seed(1)
    env = PendulumEnv()
    env.reset()
    done = False
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = TD3(state_dim, action_dim, args)
    ep_reward = 0
    time_step = 0
    state = env.reset()
    for i_ep in range(args.playing_step):
        time_step += 1
        action = agent.choose_action(state)    # action is array
        n_state, reward, done, _ = env.step(action)  # env.step accept array or list
        agent.memory.add(state, action, reward, n_state, done)
        agent.UpdateQ()
        ep_reward += reward
        if done == True or time_step == args.episode_length:
            state = env.reset()
            print(ep_reward)
            ep_reward = 0
            time_step = 0
        else:
            state = n_state

            # print('obs: {}; reward: {}'.format(observation, reward))
    # agent.save(model_path + args.filename)

