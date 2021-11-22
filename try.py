# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 21:00
# @Author  : Weiming Mai
# @FileName: try.py
# @Software: PyCharm

import gym
from Pong import AtariEnv
# env = AtariEnv()
env = gym.make('Pendulum-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low,
        env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))