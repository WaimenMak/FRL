"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
Continuous version by Ian Danforth
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, env_params=None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5  # actually half the pole's length
        self.env_param = [self.length, self.masspole, self.masscart, self.gravity]
        if env_params:
            self.length, self.masspole, self.masscart, self.gravity, self.mean = env_params
            self.env_param = env_params
        self.total_mass = (self.masspole + self.masscart)

        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


from utils.Tools import try_gpu
from agents.TD3 import Actor
import os
class Arguments():
    def __init__(self):
        self.local_bc = 0 # local update memory batch size
        self.gamma = 0
        self.lr = 0
        self.action_bound = 1
        self.tau = 0
        self.policy_noise = 0 #std of the noise, when update critics
        self.std_noise = 0    #std of the noise, when explore
        self.noise_clip = 0
        self.episode_length = 500 # env._max_episode_steps
        self.eval_episode = 2
        # self.episode_length = 200  # env._max_episode_steps
        self.episode_num = 0
        self.device = try_gpu()
        self.capacity = 0
        self.env_seed = None
        self.noisy_input = False
        # self.capacity = 10000
        self.C_iter = 0

        # self.filename = "niidevalfedstd0_noicyTrue_20000_cart1_N20_M2_L20_beta0_mu0_clientnum1actor_"
        # self.filename = "centerniidstd_noisyTrue_20000_cart5_M2_clientnum5actor_"
        self.filename = "distilstd1_noicyFalse_20000_cart5_N20_M2_L20_dualTrue_reweightTrue_distepoch10_clientnum5actor_"

# model_path = '../outputs/model/lunar/'
# model_path = '../outputs/center_model/cartpole/'
model_path = '../outputs/fed_model/cartpole/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


def cart_env_config(env_seed=None, std=0):
    # env = gym.make('CartPole-v1')
    if env_seed != None:
        np.random.seed(env_seed) #0.5, 0.1, 1, 9.8
        length = 0.5 + np.random.normal(0, std)  # length
        length = np.clip(length, 0.1, 5)
        masspole = 0.1 + np.random.normal(0, std)  # masspole
        masspole = np.clip(masspole, 0.01, 5)
        masscart = 1. + np.random.normal(0, std)  # masscart
        masscart = np.clip(masscart, 0.2, 5)
        gravity = 9.8 + np.random.normal(0, std)  # gravity
        mean = np.random.uniform(-1, 1)
        env_params = [length, masspole, masscart, gravity, mean]
        env = ContinuousCartPoleEnv(env_params)
    else:
        env = ContinuousCartPoleEnv()
    return env

if __name__ == '__main__':
    args = Arguments()
    # env = BipedalWalker()
    env = cart_env_config(4, std=0)
    print(f"mean{env.mean}")
    # print(f"r:{env.r},top:{env.stairfreqtop}")
    env.seed(5)
    print(env.env_param)
    # env.reset()
    done = False
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = Actor(state_dim, action_dim, 1, args)

    agent.load(model_path+args.filename)
    for i in range(2):
        state = env.reset()
        if args.noisy_input:
            state = state + np.random.normal(env.mean, 0.01, state.shape[0])
        ep_reward = 0
        for iter in range(args.episode_length):

            env.render()
            action = agent.predict(state)  # action is array
            # action = env.action_space.sample()
            n_state, reward, done, _ = env.step(action)  # env.step accept array or list
            if args.noisy_input:
                n_state = n_state + np.random.normal(env.mean, 0.01, state.shape[0])
            # print(reward)
            ep_reward += reward
            if done == True:
                break
            state = n_state
        print(ep_reward)
        env.close()