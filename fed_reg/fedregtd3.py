# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 19:45
# @Author  : Weiming Mai
# @FileName: fedregtd3.py
# @Software: PyCharm

# from models.Network import mlp_policy, mlp_value
from models.Network import mlp_policy, distill_qnet as mlp_value
from copy import deepcopy
from utils.Memory import replay_buffer
from utils.Tools import try_gpu
import torch.optim as optim
from torch import nn
import torch
import numpy as np


def l2_norm(local, glob):
    device = try_gpu()
    l2_loss = 0.
    for param1, param2 in zip(local.parameters(), glob.parameters()):
        l2_loss += torch.sum((param1 - param2.to(device)) ** 2)

    return l2_loss

def sim(x, y):
    return torch.cosine_similarity(x, y)

def l_con_q(state, action, local, glob, prev):
    tau = 1
    lcon_loss1 = -torch.log(torch.exp(sim(local.R_t1(state, action), glob.R_t1(state, action))/tau)/(torch.exp(sim(local.R_t1(state, action), glob.R_t1(state, action))/tau) + torch.exp(sim(local.R_t1(state, action), prev.R_t1(state, action))/tau)))

    lcon_loss2 = -torch.log(torch.exp(sim(local.R_t2(state, action), glob.R_t2(state, action))/tau)/(torch.exp(sim(local.R_t2(state, action), glob.R_t2(state, action))/tau) + torch.exp(sim(local.R_t2(state, action), prev.R_t2(state, action))/tau)))

    return torch.mean(lcon_loss1 + lcon_loss2)


def l_con_mu(state, local, glob, prev):
    tau = 1
    lcon_loss = -torch.log(torch.exp(sim(local.R_t(state), glob.R_t(state))/tau) / (
    torch.exp(sim(local.R_t(state), glob.R_t(state))/tau) + torch.exp(
        sim(local.R_t(state), prev.R_t(state))/tau)))

    return torch.mean(lcon_loss)

class Actor():
    def __init__(self, state_dim, action_dim, args):
        self.action_bound = args.action_bound
        self.action_dim = action_dim
        self.device = args.device
        self.std_noise = args.action_bound * args.std_noise #std of the noise, when explore
        self.std_policy_noise = args.policy_noise     #std of the noise, when update critics
        self.noise_clip = args.noise_clip
        self.policy_net = mlp_policy(state_dim, action_dim, self.action_bound)
        self.target_net = mlp_policy(state_dim, action_dim, self.action_bound)
        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.beta = args.beta
        self.mu = args.mu
        # self.alpha = args.alpha
        self.l_mse = nn.MSELoss()
        # self.glob_mu = None
        self.glob_mu = deepcopy(self.policy_net)
        self.prev_mu = deepcopy(self.policy_net)
        self.temp_mu = deepcopy(self.policy_net)

    def predict(self, state):  # for visualize and test
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action = self.policy_net(state).numpy()

        return action

    def choose_action(self, state):
        # for exploration
        # state: 1 * state_dim
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            action = (
                self.policy_net(state).cpu().numpy() + np.random.normal(0, self.std_noise, size=self.action_dim)
            ).clip(-self.action_bound, self.action_bound)  # constraint action bound

        return action

    def choose_action2(self, state):
        # for update Qs on gpu
        # state: bc * state_dim
        with torch.no_grad():
            # state = torch.tensor(state, device=self.device, dtype=torch.float32)
            noise = torch.tensor(np.random.normal(0, self.std_policy_noise, size=[state.size(0), self.action_dim]).clip(
                    -self.noise_clip,self.noise_clip), dtype=torch.float).to(self.device)
            action = (
                self.target_net(state) + noise            # noise is tensor on gpu
            ).clip(-self.action_bound, self.action_bound)  # constraint action bound

        return action

    def update_policy(self, state, Q_net):
        self.temp_mu.load_state_dict(self.policy_net.state_dict())
        # if self.alpha != 0:
        #     actor_loss = -Q_net.Q1_val(state, self.policy_net(state)).mean() + self.alpha * self.l_mse(self.policy_net(state), self.glob_mu(state))
        if self.beta !=0:
            actor_loss = -Q_net.Q1_val(state, self.policy_net(state)).mean() + self.beta * l2_norm(self.policy_net, self.glob_mu)
        elif self.mu !=0:
            actor_loss = -Q_net.Q1_val(state, self.policy_net(state)).mean() + self.mu * l_con_mu(state, self.policy_net, self.glob_mu, self.prev_mu)
        else:
            actor_loss = -Q_net.Q1_val(state, self.policy_net(state)).mean()
        # print(f'actor loss{actor_loss:.2f}')
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


    def update_target(self, tau, mu_params):
        for params in mu_params.keys():
            self.target_net.state_dict()[params].copy_(tau * mu_params[params] + (1 - tau) * self.target_net.state_dict()[params])

    def save(self, PATH):
        torch.save(self.policy_net.state_dict(), PATH + "actor_td3.pth")

    def load(self, PATH):
        self.policy_net.load_state_dict(torch.load(PATH + "actor_td3.pth"))
        self.policy_net.cpu()

class Critic():
    def __init__(self,action_dim, state_dim, args):
        self.Q_net = mlp_value(state_dim, action_dim)
        self.Q_target = mlp_value(state_dim, action_dim)

        self.critic_optimizer = optim.Adam(self.Q_net.parameters(), lr=args.lr)

    def predict(self, state, action):
        q_val1, q_val2 = self.Q_net(state, action)
        return q_val1, q_val2

    def target(self, state, action):
        q_val1, q_val2 = self.Q_target(state, action)
        return q_val1, q_val2

    def update_critics(self):
        pass

    def update_target(self, tau, q_params):
        for params in q_params.keys():
            self.Q_target.state_dict()[params].copy_(tau * q_params[params] + (1 - tau) * self.Q_target.state_dict()[params])

class fedTD3():
    def __init__(self, state_dim, action_dim, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = None
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = args.device
        # self.C_iter = args.C_iter
        self.iter = 0   # actor policy send frequency
        self.count = 0
        self.L = args.L
        self.memory = replay_buffer(args.capacity)
        self.batch_size = args.local_bc
        self.actor = Actor(state_dim, action_dim, args)
        self.critic = Critic(state_dim, action_dim, args)
        # self.actor_loss = Critic.Q1_net.forward()
        self.glob_q = deepcopy(self.critic.Q_net)
        self.temp_q = deepcopy(self.critic.Q_net)
        self.prev_q = deepcopy(self.critic.Q_net)

        self.beta = args.beta
        self.mu = args.mu
        self.dual = args.dual
        # self.alpha = args.alpha
        self.l_mse = nn.MSELoss()
        self.critics_loss = nn.MSELoss()

    def UpdateQ(self):
        if len(self.memory) < self.batch_size:
            return
        # self.iter += 1
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float) #bc * state_dim
        action_batch = torch.tensor(
            action_batch, device=self.device, dtype=torch.float)  # bc * action_dim
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float).view(-1, 1)
        n_state_batch = torch.tensor(
            n_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device, dtype=torch.float).view(-1, 1)

        self.temp_q.load_state_dict(self.critic.Q_net.state_dict())
        with torch.no_grad():
            # action_tilde = self.actor.choose_action2(state_batch)
            action_tilde = self.actor.choose_action2(n_state_batch)  #next_action
            q1_target, q2_target = self.critic.target(n_state_batch, action_tilde)

            max_target_q_val = torch.cat((q1_target, q2_target), dim=1).min(1)[0].detach().view(-1, 1)
            y_hat = reward_batch + self.gamma * max_target_q_val * (1 - done_batch)

        current_q_val = self.critic.predict(state_batch, action_batch)
        if self.beta != 0:
            # loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1],y_hat) + self.alpha * self.l_mse(torch.cat(self.critic.Q_net(state_batch, action_batch)), torch.cat(self.glob_q(state_batch, action_batch)))
            loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1],y_hat) + self.beta * l2_norm(self.critic.Q_net, self.glob_q)
        elif self.mu != 0:
            loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1], y_hat) + self.mu * l_con_q(state_batch, action_batch, self.critic.Q_net, self.glob_q, self.prev_q)
        else:
            loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1], y_hat)
        # print(f'critic loss{loss:.2f}')
        self.critic.critic_optimizer.zero_grad()
        loss.backward()
        self.critic.critic_optimizer.step()

        self.prev_q.load_state_dict(self.temp_q.state_dict())
        # if self.iter % args.M == 0:
        #     self.localDelayUpdate(state_batch, self.critic.Q_net, self.tau, client_pipe)
        return state_batch

    # def DelayUpdate(self, state, Q_net, tau):
    #     """
    #     :param state:  state batch from UpdateQ()
    #     :param Q_net:  critic.Qnet
    #     :return:
    #     """
    #     self.actor.update_policy(state, Q_net)
    #     self.actor.update_target(tau)
    #     self.critic.update_target(tau)
    def dual_distill(self, state):
        alpha = 0.5
        with torch.no_grad():
            V1 = self.glob_q.Q1_val(state, self.actor.glob_mu(state))  #action batch
            V2 = self.critic.Q_net.Q1_val(state, self.actor.policy_net(state))
        loss = torch.sum((self.actor.glob_mu(state) - self.actor.policy_net(state))**2 * alpha * torch.exp(V1 - V2).view(-1,1)).mean()

        return loss

    def localDelayUpdate(self, state, Q_net, tau, client_pipe):
        """
        :param state:  state batch from UpdateQ()
        :param Q_net:  critic.Qnet
        :return: 
        """
        self.count += 1
        if self.dual:
            partial = 0.9
            loss2 = 0.
            loss1 = -Q_net.Q1_val(state, self.actor.policy_net(state)).mean()
            if self.count > self.L:
                loss2 = self.dual_distill(state)

            distil_loss = partial * loss1 + (1 - partial) * loss2
            self.actor.actor_optimizer.zero_grad()
            distil_loss.backward()
            self.actor.actor_optimizer.step()
        else:
            self.actor.update_policy(state, Q_net)

        self.actor.prev_mu.load_state_dict(self.actor.temp_mu.state_dict())

        if self.count % self.L == 0:
            models = [self.actor.policy_net, self.actor.target_net, self.critic.Q_target, self.critic.Q_net]
            self.to_cpu(models)
            client_pipe.send((self.actor.policy_net.state_dict(), True))
            mu_params, q_params = client_pipe.recv()
            self.actor.glob_mu.cpu()
            self.actor.glob_mu.load_state_dict(mu_params)
            self.actor.glob_mu.to(self.device)
            # self.glob_q = q_params
            self.glob_q.cpu()
            self.glob_q.load_state_dict(q_params)
            self.glob_q.to(self.device)
            self.actor.policy_net.load_state_dict(mu_params) #local mu = mu agg
            # for param in (mu_params.keys()):
            #     self.actor.policy_net.state_dict()[param].copy_(mu_params[param]) #agg

            self.actor.update_target(tau, mu_params)
            self.critic.update_target(tau, q_params)
            self.to_gpu(models)
            return
    #
        self.actor.update_target(tau, self.actor.policy_net.state_dict())
        self.critic.update_target(tau, self.critic.Q_net.state_dict())

    def sync(self, q_params, mu_params):
        self.critic.Q_net.load_state_dict(q_params)
        self.critic.Q_net.to(self.device)
        self.glob_q.load_state_dict(q_params)
        self.prev_q.load_state_dict(q_params)
        self.glob_q.to(self.device)
        self.critic.Q_target.load_state_dict(q_params)
        self.critic.Q_target.to(self.device)

        self.actor.policy_net.load_state_dict(mu_params)
        self.actor.policy_net.to(self.device)
        # self.actor.glob_mu = mu_params
        self.actor.glob_mu.load_state_dict(mu_params)
        self.actor.prev_mu.load_state_dict(mu_params)
        self.actor.glob_mu.to(self.device)
        self.actor.target_net.load_state_dict(mu_params)
        self.actor.target_net.to(self.device)

        self.to_gpu([self.temp_q, self.prev_q, self.actor.temp_mu, self.actor.prev_mu])

    def to_cpu(self, models):
        for model in models:
            model.cpu()

    def to_gpu(self, models):
        for model in models:
            model.to(self.device)

    def choose_action(self, state):
        action = self.actor.choose_action(state)
        return action

    def predict(self, state): # for eval
        with torch.no_grad():
            # state = torch.tensor(state, dtype=torch.float).cuda()
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.actor.policy_net(state).cpu().numpy()

        return action

    def save(self, PATH):
        torch.save(self.actor.policy_net.state_dict(), PATH + "actor_td3.pth")
        torch.save(self.critic.Q_net.state_dict(), PATH + "critic_td3.pth")


    def load(self, PATH):
        self.actor.policy_net.load_state_dict(torch.load(PATH + 'td3.pth'))
        self.actor.target_net.load_state_dict(self.actor.policy_net.state_dict())
        self.critic.Q_net.load_state_dict(torch.load(PATH + 'critic_td3.pth'))
        self.critic.Q_target.load_state_dict(self.critic.Q_net.state_dict())

