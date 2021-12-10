# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 21:00
# @Author  : Weiming Mai
# @FileName: try.py
# @Software: PyCharm

import gym
env = gym.make('BipedalWalker-v3')
env.reset()
done = False
# env.ale.getAvailableDifficulties()
for i in range(30):
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    print('obs: {}; reward: {}'.format(observation, reward))
    env.render()
#     time.sleep(0.1)
env.close()