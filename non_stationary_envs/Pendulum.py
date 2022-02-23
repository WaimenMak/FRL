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
            self.l, self.m, self.g, self.mean = env_params
            self.env_param = env_params
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

# def pendulum_env_config(env_seed):
#     # env = gym.make('Pendulum-v1')
#     if env_seed != None:
#         np.random.seed(env_seed) #0.5, 0.1, 1, 9.8
#         length = np.random.uniform(0.1, 3, 1)[0]  # length
#         masspole = np.random.uniform(0.1, 2.5, 1)[0]  # masspole
#         # torque = np.random.uniform(1, 3, 1)[0]  # masscart
#         gravity = np.random.uniform(7, 13, 1)[0]  # gravity
#         env_params = [length, masspole, gravity]
#         env = PendulumEnv(env_params)
#     else:
#         env = PendulumEnv()
#     return env

def pendulum_env_config2(env_seed=None, std=0):
    # env = gym.make('Pendulum-v1')
    if env_seed != None:
        np.random.seed(env_seed) #0.5, 0.1, 1, 9.8
        length = 1. + np.random.normal(0, std)  # length
        length = np.clip(length, 0.5, 1.5)
        masspole = 1. + np.random.normal(0, std)  # masspole
        masspole = np.clip(masspole, 0.5, 1.5)
        gravity = 10. + np.random.normal(0, std)  # gravity
        gravity = np.clip(gravity, 8, 12)
        mean = np.random.uniform(-1, 1)
        env_params = [length, masspole, gravity, mean]
        env = PendulumEnv(env_params)
    else:
        env = PendulumEnv()
    return env

from agents.TD3 import Actor
import os
class Arguments():
    def __init__(self):
        # self.local_bc = 256  # local update memory batch size
        # self.gamma = 0.98
        self.lr = 0.002
        # self.action_bound = 2
        self.tau = 0.01
        self.policy_noise = 0 #std of the noise, when update critics
        self.std_noise = 0    #std of the noise, when explore
        self.noise_clip = 0

        self.env_name = "pendulum"

        if self.env_name == "pendulum":
            self.action_bound = 2
            self.local_bc = 0  # local update memory batch size
            self.episode_length = 200  # env._max_episode_steps
            self.playing_step = int(2e4)
            self.capacity = 10
            self.std = 0
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
        # self.capacity = 10000
        # self.episode_length = 200  # env._max_episode_steps
        self.eval_episode = 100
        self.filename = "niidevalfedstd1_noicyFalse_32000_pendulum5_N100_M2_L100_beta0_mu0.01_dual_False_clientnum5actor_"  #moon
        # self.filename = "distilstd1_noicyFalse_32000_pendulum5_N100_M2_L100_dualFalse_distepoch20_clientnum5actor_"  # dist


# model_path = '../outputs/center_model/pendulum/'
model_path = '../outputs/fed_model/pendulum/'


if not os.path.exists(model_path):
    os.makedirs(model_path)


if __name__ == '__main__':
    args = Arguments()
    env = pendulum_env_config2(env_seed=3, std=1)

    env.seed(1)
    # env.reset()
    done = False
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = Actor(state_dim, action_dim, args.action_bound, args)
    agent.load(model_path+args.filename)
    for i in range(2):
        state = env.reset()
        ep_reward = 0

        for i_ep in range(args.eval_episode):
            env.render()
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
        print(ep_reward)
        env.close()