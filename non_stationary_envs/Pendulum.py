# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 22:02
# @Author  : Weiming Mai
# @FileName: Pendulum.py
# @Software: PyCharm

from os import path
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
from utils.Tools import try_gpu, action_trans
from agents.DQN import DQN


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, env_params=None, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        # self.l = 2.
        self.viewer = None
        if env_params:
            self.l, self.m, self.g = env_params

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def pendulum_env_config(env_seed):
    # env = gym.make('Pendulum-v1')
    if env_seed != None:
        np.random.seed(env_seed) #0.5, 0.1, 1, 9.8
        length = np.random.uniform(0.1, 3, 1)[0]  # length
        masspole = np.random.uniform(0.1, 2.5, 1)[0]  # masspole
        # torque = np.random.uniform(1, 3, 1)[0]  # masscart
        gravity = np.random.uniform(7, 13, 1)[0]  # gravity
        env_params = [length, masspole, gravity]
        env = PendulumEnv(env_params)
    else:
        env = PendulumEnv()
    return env

def pendulum_env_config2(env_seed):
    # env = gym.make('Pendulum-v1')
    if env_seed != None:
        np.random.seed(env_seed) #0.5, 0.1, 1, 9.8
        length = 1. + np.random.normal(0, 0.5) #length
        masspole = 1. + np.random.normal(0, 0.5)  # masspole
        # torque = np.random.uniform(1, 3, 1)[0]  # masscart
        gravity = 10. + np.random.normal(0, 0.5)  # gravity
        env_params = [length, masspole, gravity]
        env = PendulumEnv(env_params)
    else:
        env = PendulumEnv()
    return env

if __name__ == '__main__':
    device = try_gpu()
    # model_path = '../outputs/model/pendulum/'
    model_path = '../outputs/fed_model/'
    # env = CartPoleEnv()
    env = pendulum_env_config(7)
    print(f"l:{env.l:.2f},g: {env.g:.2f},m:{env.m:.2f}")
    upperbound = env.action_space.low[0]
    lowerbound = env.action_space.high[0]
    # env.seed = 1
    # agent = fed_DQN(env.observation_space.shape[0], env.action_space.n, 1, 1,
    #                       1, 1, 1, device)
    action_dim = 11
    agent = DQN(env.observation_space.shape[0], action_dim, 1, 1,
                    1, 1, 1, device)
    # agent.name = 'agent_server'
    agent.load(model_path + 'clients_5_Pendulum_fedavgdqn.pth')
    # agent.load(model_path + 'seed_0_pend_feddqn.pth')
    # agent.load(model_path + 'seed_0_dqn_pendulum.pth')
    state = env.reset()  # 初始化环境，observation为环境状态
    count = 0
    ep_reward = 0
    for _ in range(200):
        env.render()
        # action = env.action_space.sample()  # 随机采样动作
        # print(action)
        action = action_trans(agent.predict(state), action_dim, upperbound, lowerbound)
        n_state, reward, done, _ = env.step([action])
        ep_reward += reward
        state = n_state
        if done:
            break
        # count += 1
        time.sleep(0.1)
    env.close()
    print(ep_reward)