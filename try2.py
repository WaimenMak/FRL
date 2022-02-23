# -*- coding: utf-8 -*-
# @Time    : 2021/12/4 19:42
# @Author  : Weiming Mai
# @FileName: walker.py
# @Software: PyCharm
import sys
import math

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle



FPS    = 50


SCALE  = 30.0
MOTORS_TORQUE = 80
SPEED_HIP     = 4
SPEED_KNEE    = 6
LIDAR_RANGE   = 160/SCALE

INITIAL_RANDOM = 5

HULL_POLY =[
    (-30,+9), (+6,+9), (+34,+1),
    (+34,-8), (-30,-8)
    ]
LEG_DOWN = -8/SCALE
LEG_W, LEG_H = 8/SCALE, 34/SCALE
# LEG_W, LEG_H = 8/SCALE, 10/SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5

HULL_FD = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in HULL_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0020,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy

LEG_FD = fixtureDef(
                    shape=polygonShape(box=(LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

LOWER_FD = fixtureDef(
                    shape=polygonShape(box=(0.8*LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.hull==contact.fixtureA.body or self.env.hull==contact.fixtureB.body:
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True
    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False

import random
def select(r):
    state = range(1, 4)
    # 概率列表
    # r = [1/4, 1/4, 1/4, 1/4]
    # print(r)
    sum = 0
    ran_num = random.random()
    for state, r in zip(state, r):
        sum += r
        if ran_num < sum:
            break
    return state

class BipedalWalker(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    hardcore = False

    def __init__(self, seed=None):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        # self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None

        self.prev_shaping = None

        # if not seed:
        self.stairfreqtop = 5  # 5
        self.stairfreqlow = 3
        self.r = [1/3, 1/3, 1/3]
        if seed:
            np.random.seed(seed)
            top = [2, 5, 10]
            # self.stairfreqtop = np.random.randint(1, 10)
            self.stairfreqtop = np.random.choice(top)
            self.stairfreqlow = 1
            self.r = np.random.dirichlet(np.ones(3)).tolist()  #freq
            self.env_param = self.r

    def modify(self, seed):
        if seed:
            np.random.seed(seed)
            top = [2, 5, 10]
            # self.stairfreqtop = np.random.randint(1, 10)
            self.stairfreqtop = np.random.choice(top)
            self.stairfreqlow = 1
            self.r = np.random.dirichlet(np.ones(3)).tolist()  # freq
            self.env_param = self.r


        # self.fd_polygon = fixtureDef(
        #                 shape = polygonShape(vertices=
        #                 [(0, 0),
        #                  (1, 0),
        #                  (1, -1),
        #                  (0, -1)]),
        #                 friction = FRICTION)
        #
        # self.fd_edge = fixtureDef(
        #             shape = edgeShape(vertices=
        #             [(0, 0),
        #              (1, 1)]),
        #             friction = FRICTION,
        #             categoryBits=0x0001,
        #         )
        #
        # # self.reset()
        #
        # high = np.array([np.inf] * 24)
        # self.action_space = spaces.Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]), dtype=np.float32)
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)



class BipedalWalkerHardcore(BipedalWalker):
    hardcore = True

def walker_config(env_seed=None):
    if env_seed != None:
        env = BipedalWalkerHardcore(env_seed)
    else:
        env = BipedalWalkerHardcore()

    return env


from utils.Tools import try_gpu
from non_stationary_envs.Pendulum import PendulumEnv
import os


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

def walker_config(env_seed=None):
    if env_seed != None:
        env_param =0.45
        env = BipedalWalkerHardcore(env_seed, env_param)
    else:
        env = BipedalWalkerHardcore()

    return env

from torch.multiprocessing import Pipe, Process
from threading import Thread
def aa(aaa):
    print(aaa.env_param)

def aab(pipe, aaa):
    seed = pipe.recv()
    aaa.modify(seed)
    print(aaa.env_param)
if __name__ == '__main__':
    process_num = 2
    # env = gym.make('MountainCarContinuous-v0')
    # serving = agent()
    # serving.name = "server"
    # agent1 = agent()
    # agent1.name = 1
    # agent2 = agent()
    # agent2.name = 2
    # pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num) for pipe1, pipe2 in (Pipe(),))
    # aaa = BipedalWalkerHardcore(seed=1)
    # aaa = pendulum_env_config2(1, 0.2)
    pipe1, pipe2 = Pipe()
    aaa = BipedalWalkerHardcore(1)
    # p = Thread(target=server, args=(pipe_dict, process_num, serving))
    # p = Thread(target=ss, args=(pipe1, aaa))
    # p2 = Process(target=client, args=(ss,pipe_dict[0][0]))
    # p3 = Process(target=aa, args=(pipe2, aaa))
    p = Thread(target=aa, args=(aaa,))
    p3 = Process(target=aab, args=(pipe2,aaa))

    p.start()
    p3.start()
    pipe1.send(1)
    [oo.join() for oo in [p, p3]]
    print(aaa.env_param)