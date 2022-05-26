# -*- coding: utf-8 -*-
# @Time    : 2022/5/21 14:19
# @Author  : Weiming Mai
# @FileName: imporve_actor.py
# @Software: PyCharm

from disttd3v3 import Actor
from non_stationary_envs.walker import BipedalWalkerHardcore
from utils.Tools import try_gpu
from utils.Memory import replay_buffer
from models.Network import distill_qnet
from scipy import stats
import torch
import numpy as np

class Arguments():
    def __init__(self):
        self.local_bc = 256  # local update memory batch size
        self.gamma = 0.98
        self.lr = 0.002
        self.action_bound = 1
        self.tau = 0.01
        self.policy_noise = 0.01 #std of the noise, when update critics
        self.std_noise = 0.01    #std of the noise, when explore
        self.noise_clip = 0.5
        self.episode_length = 1600 # env._max_episode_steps
        self.eval_episode = 2
        # self.episode_length = 200  # env._max_episode_steps
        self.episode_num = 2000
        self.device = try_gpu()
        self.capacity = 1e6
        self.env_seed = None
        # self.capacity = 10000
        self.C_iter = 5
        self.beta = 0
        self.mu = 0
        self.alpha = 0
        # self.filename = "v3_distilstd0_noicyFalse_1200000_walker5_N400_M2_L400_criticdualTrue0.1_actordualFalse0.9_reweightFalse1_distepoch20_lrdecayFalse_actor_1actor_"  #best walker dist
        #best walker in walker v1
        #baseline best
        # self.filename = "v3_distilstd0_noicyFalse_1200000_walker5_N400_M2_L400_criticdualTrue0.1_epc40_lr0.002_actordualFalse0.9_reweightFalse1_distepoch20_lrdecayFalse_actor_4actor_"
        #sccd
        self.filename = "fedprox"  #sccd:score 288 in v3walker, fedagv: score 258 in v1 walker, moon score 259 in v1,fedprox 180+ in v1

# model_path = '../outputs/fed_model/walker/'
model_path = '../outputs/fed_model/Q_value/'
if __name__ == '__main__':
    args = Arguments()
    # env = BipedalWalker()
    # env = BipedalWalkerHardcore()
    env = BipedalWalkerHardcore()
    # env.seed(1)

    print(f"r:{env.r},top:{env.stairfreqtop}")

    # env2.seed(1)
    # env.reset()
    done = False
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    group1 = []
    group2 = []
    agent = Actor(state_dim, action_dim, args)
    rollout = replay_buffer(args.capacity)
    # agent = Actor(state_dim, action_dim, args)
    # policy_state_dict = torch.load(f"../outputs/fed_model/Q_value/{args.filename}.pth", map_location='cpu')
    # agent.policy_net.to("cpu")
    # agent.policy_net.load_state_dict(policy_state_dict)

    model_q = distill_qnet(state_dim, action_dim)
    model = torch.load(f"../outputs/fed_model/Q_value/agent0_Q_model_{args.filename}.pth", map_location='cpu')
    model_q.load_state_dict(model)
    policy_state_dict = torch.load(f"../outputs/fed_model/Q_value/{args.filename}_actor.pth", map_location='cpu')
    agent = Actor(state_dim, action_dim, args)
    for trial in range(30):
        env = BipedalWalkerHardcore()
        env.seed(1)
        rollout.clear()
        # agent = Actor(state_dim, action_dim, args)
        rollout = replay_buffer(args.capacity)
        # agent = Actor(state_dim, action_dim, args)
        # policy_state_dict = torch.load(f"../outputs/fed_model/Q_value/{args.filename}_actor.pth", map_location='cpu')
        # agent.policy_net.to("cpu")
        agent.policy_net.load_state_dict(policy_state_dict)

        for i in range(1):
            state = env.reset()
            # state = env.reset_notchange()
            ep_reward = 0
            for iter in range(args.episode_length):

                # env.render()
                action = agent.predict(state)  # action is array
                n_state, reward, done, _ = env.step(action)  # env.step accept array or list
                rollout.add(state, action, reward, n_state, done)
                # print(reward)
                ep_reward += reward
                if done == True:
                    break
                state = n_state
            group1.append(ep_reward)
            print(ep_reward)
            # env.close()
###########################
        # model_q = distill_qnet(state_dim, action_dim)
        # model = torch.load(f"../outputs/fed_model/Q_value/agent0_Q_model_{args.filename}.pth", map_location='cpu')
        model_q.load_state_dict(model)

        # for i in range(1):
        #     state = env.reset_notchange()
        #     ep_reward = 0
        #     for iter in range(args.episode_length):
        #
        #         env.render()
        #         # action = agent.predict(state)  # action is array
        #         s_tilde = torch.tensor(np.tile(state, [300, 1]), dtype=torch.float)
        #         a_tilde = torch.tensor(-1 + 2 * np.random.random((300, 4)), dtype=torch.float)
        #         q_tilde = model_q.Q1_val(s_tilde, a_tilde).detach().numpy()
        #         action = a_tilde[np.argmax(q_tilde)].numpy()
        #         n_state, reward, done, _ = env.step(action)  # env.step accept array or list
        #         rollout.add(state, action, reward, n_state, done)
        #         # print(reward)
        #         ep_reward += reward
        #         if done == True:
        #             break
        #         state = n_state
        #     print(ep_reward)
        #     env.close()

        states, _, _, _, _ = rollout.sample(len(rollout))
        states = torch.tensor(np.array(states), dtype=torch.float)

        for epc in range(1):
            agent.update_policy(states, model_q)
        for i in range(1):
            state = env.reset_notchange()
            ep_reward = 0
            for iter in range(args.episode_length):

                # env.render()
                action = agent.predict(state)
                n_state, reward, done, _ = env.step(action)  # env.step accept array or list
                rollout.add(state, action, reward, n_state, done)
                # print(reward)
                ep_reward += reward
                if done == True:
                    break
                state = n_state
            group2.append(ep_reward)
            print(ep_reward)
        # env.close()

    print(stats.ttest_ind(group1, group2))
    group1 = np.array(group1)
    group2 = np.array(group2)
    print([np.mean(group1), np.std(group1), np.mean(np.abs(np.array(group1)+np.min(group1) - np.mean(group1+np.min(group1))))])
    print([np.mean(group2), np.std(group2), np.mean(np.abs(np.array(group2)+np.min(group2) - np.mean(group2+np.min(group2))))])
