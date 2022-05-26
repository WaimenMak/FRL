# -*- coding: utf-8 -*-
# @Time    : 2022/5/5 21:29
# @Author  : Weiming Mai
# @FileName: sccd_conv.py
# @Software: PyCharm


from models.Conv_model import conv_policy, distill_qnet as conv_value
from copy import deepcopy
from utils.Memory import replay_buffer
import torch.optim as optim
from torch import nn
import torch
import numpy as np


def l2_norm(local, glob):
    l2_loss = 0.
    for param1, param2 in zip(local.parameters(), glob.parameters()):
        l2_loss += torch.sum((param1 - param2.cuda()) ** 2)

    return l2_loss

def KL(local, glob, states, actions, softmax, n):
    tau = 0.5
    actions = actions.view(-1, 1)
    loc = local(states, actions)
    glo = glob(states, actions)

    p_new1 = softmax(loc[0].view(states.size(0) // n, -1) / tau)
    p_old1 = softmax(glo[0].view(states.size(0) // n, -1) / tau)

    p_new2 = softmax(loc[1].view(states.size(0) // n, -1) / tau)
    p_old2 = softmax(glo[1].view(states.size(0) // n, -1) / tau)

    KL_loss1 = torch.sum(p_old1 * torch.log(p_old1/(p_new1 + torch.tensor(1e-8))))
    KL_loss2 = torch.sum(p_old2 * torch.log(p_old2/(p_new2 + torch.tensor(1e-8))))

    # KL_loss1 = torch.sum(p_new1 * torch.log(p_new1/(p_old1 + torch.tensor(1e-8))))
    # KL_loss2 = torch.sum(p_new2 * torch.log(p_new2/(p_old2 + torch.tensor(1e-8))))
    return KL_loss1 + KL_loss2

class Actor():
    def __init__(self, state_dim, action_dim, args):
        self.action_bound = args.action_bound
        self.action_dim = action_dim
        self.device = args.device
        self.std_noise = args.action_bound[1, 1] * args.std_noise  # std of the noise, when explore
        self.cov1 = np.array([[self.std_noise, 0, 0], [0, self.std_noise / 2, 0], [0, 0, self.std_noise / 2]])
        self.std_policy_noise = args.policy_noise  # std of the noise, when update critics
        self.cov2 = np.array(
            [[self.std_policy_noise, 0, 0], [0, self.std_policy_noise / 2, 0], [0, 0, self.std_policy_noise / 2]])
        self.noise_clip = args.noise_clip
        self.policy_net = conv_policy(4, action_dim, self.action_bound)
        self.target_net = conv_policy(4, action_dim, self.action_bound)
        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.beta = args.beta
        self.mu = args.mu
        self.alpha = args.alpha
        # self.choice = args.choice
        self.l_mse = nn.MSELoss()
        # self.glob_mu = None
        self.glob_mu = deepcopy(self.policy_net)


    def predict(self, state):  # for visualize and test
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            state = state.permute(2, 0, 1).unsqueeze(0)
            action = self.policy_net(state).numpy().squeeze()

        return action

    def choose_action(self, state):
        # for exploration
        # state: 1 * state_dim, 96*96*4

        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            # state = state.permute(2, 0, 1).unsqueeze(0)  # 1*4*96*96
            state = state.unsqueeze(0)  # 1*4*96*96
            action = self.policy_net(state).cpu().numpy() + np.random.multivariate_normal((0, 0.5, 0), self.cov1, size=1) #0.5 make the noise>0
            # constraint action bound
            action = np.minimum(np.maximum(action, self.action_bound[0].cpu().numpy()), self.action_bound[1].cpu().numpy())
            action = np.squeeze(action)

        return action

    def choose_action2(self, state):
        # for update Qs on gpu
        # state: bc * state_dim, bc * 96*96*4
        with torch.no_grad():
            # state = torch.tensor(state, device=self.device, dtype=torch.float32)
            # state = state.permute(0, 3, 1, 2)
            noise = torch.tensor(np.random.multivariate_normal((0, 0.5, 0), self.cov2, size=[state.size(0)]).clip(
                    -self.noise_clip,self.noise_clip), dtype=torch.float).to(self.device)
            action = self.target_net(state) + noise            # noise is tensor on gpu
            if self.action_bound.device != action.device:
                self.action_bound = self.action_bound.to(action.device)

            action = torch.min(torch.max(action, self.action_bound[0]), self.action_bound[1])

        return action

    def update_policy(self, state, Q_net):
        # self.temp_mu.load_state_dict(self.policy_net.state_dict())
        # if self.choice == "mse":
        #     actor_loss = -Q_net.Q1_val(state, self.policy_net(state)).mean() + self.alpha * self.l_mse(self.policy_net(state), self.glob_mu(state))
        # elif self.choice == "KL":
        #     actor_loss = -Q_net.Q1_val(state, self.policy_net(state)).mean()

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
    def __init__(self,state_dim, action_dim, args):
        self.Q_net = conv_value(4, action_dim)  #4: inpt channel
        self.Q_target = conv_value(4, action_dim)

        self.critic_optimizer = optim.Adam(self.Q_net.parameters(), lr=args.lr)

    def predict(self, state, action):
        q_val1, q_val2 = self.Q_net(state, action)
        return q_val1, q_val2

    def target(self, state, action):
        q_val1, q_val2 = self.Q_target(state, action)
        return q_val1, q_val2

    def partial_update(self, tau, q_params):  # q_params only the output layer
        for params in q_params[0].keys():
            self.Q_target.oupt_layer_q1.state_dict()[params].copy_(tau * q_params[0][params] + (1 - tau) * self.Q_target.oupt_layer_q1.state_dict()[params])
        for params in q_params[1].keys():
            self.Q_target.oupt_layer_q2.state_dict()[params].copy_(tau * q_params[1][params] + (1 - tau) * self.Q_target.oupt_layer_q2.state_dict()[params])

    def update_target(self, tau, q_params): #q_params: whole state dict of Q net
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
        # self.distil_opt = optim.Adam(self.critic.Q_net.parameters(), lr=args.critic_lr)
        # self.distil_opt = optim.Adam(self.critic.Q_net.parameters(), lr=args.critic_lr, weight_decay=1e-2)
        # self.actor_loss = Critic.Q1_net.forward()
        self.glob_q = deepcopy(self.critic.Q_net)
        # self.temp_q = deepcopy(self.critic.Q_net)
        # self.prev_q = deepcopy(self.critic.Q_net)

        self.beta = args.beta
        self.mu = args.mu
        self.alpha = args.alpha
        self.actor_dual = args.actor_dual
        self.actor_partial = args.actor_partial
        self.actor_epc = args.actor_epc
        # self.choice = args.choice
        # self.noise_clip = args.noise_clip
        # self.action_bound = args.action_bound
        self.l_mse = nn.MSELoss()
        # self.softmax = nn.Softmax(dim=1)
        # self.n = 10
        self.critics_loss = nn.MSELoss()

    def critic_distill(self, partial, epc):
        # state_batch, action_batch, reward_batch, n_state_batch, done_batch = self.memory.distil_sample(
        #     self.batch_size, epc)
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(
            np.array(state_batch), device=self.device, dtype=torch.float) #bc * state_dim
        action_batch = torch.tensor(
            np.array(action_batch), device=self.device, dtype=torch.float)  # bc * action_dim
        reward_batch = torch.tensor(
            np.array(reward_batch), device=self.device, dtype=torch.float).view(-1, 1)
        n_state_batch = torch.tensor(
            np.array(n_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device, dtype=torch.float).view(-1, 1)

        # self.temp_q.load_state_dict(self.critic.Q_net.state_dict())
        with torch.no_grad():
            # action_tilde = self.actor.choose_action2(state_batch)
            action_tilde = self.actor.choose_action2(n_state_batch)  #next_action
            q1_target, q2_target = self.critic.target(n_state_batch, action_tilde)

            max_target_q_val = torch.cat((q1_target, q2_target), dim=1).min(1)[0].detach().view(-1, 1)
            y_hat = reward_batch + self.gamma * max_target_q_val * (1 - done_batch)

        current_q_val = self.critic.predict(state_batch, action_batch)

        loss1 = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1], y_hat)
        loss2 = self.l_mse(torch.cat((current_q_val[0], current_q_val[1])),torch.cat(self.glob_q(state_batch, action_batch)))
        # global_q_val = self.glob_q(state_batch, action_batch)
        # loss2 = self.l_mse(current_q_val[0], global_q_val[0]) + self.l_mse(current_q_val[1], global_q_val[1])

        critic_loss = partial * loss1 + (1 - partial) * loss2
        self.critic.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic.critic_optimizer.step()

        # self.distil_opt.zero_grad()
        # critic_loss.backward()
        # self.distil_opt.step()

    def UpdateQ(self):
        if len(self.memory) < self.batch_size:
            return
        # self.iter += 1
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(
            np.array(state_batch), device=self.device, dtype=torch.float)  # bc * state_dim
        action_batch = torch.tensor(
            np.array(action_batch), device=self.device, dtype=torch.float)  # bc * action_dim
        reward_batch = torch.tensor(
            np.array(reward_batch), device=self.device, dtype=torch.float).view(-1, 1)
        n_state_batch = torch.tensor(
            np.array(n_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device, dtype=torch.float).view(-1, 1)

        # self.temp_q.load_state_dict(self.critic.Q_net.state_dict())
        with torch.no_grad():
            # action_tilde = self.actor.choose_action2(state_batch)
            action_tilde = self.actor.choose_action2(n_state_batch)  #next_action
            q1_target, q2_target = self.critic.target(n_state_batch, action_tilde)

            max_target_q_val = torch.cat((q1_target, q2_target), dim=1).min(1)[0].detach().view(-1, 1)
            y_hat = reward_batch + self.gamma * max_target_q_val * (1 - done_batch)

        current_q_val = self.critic.predict(state_batch, action_batch)

        loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1], y_hat)
        self.critic.critic_optimizer.zero_grad()
        loss.backward()
        self.critic.critic_optimizer.step()

        # self.prev_q.load_state_dict(self.temp_q.state_dict())
        # if self.iter % args.M == 0:
        #     self.localDelayUpdate(state_batch, self.critic.Q_net, self.tau, client_pipe)
        return state_batch, action_batch

    def dual_distill(self, state):
        alpha = 0.01   #default 0.5
        with torch.no_grad():
            V1 = self.glob_q.Q1_val(state, self.actor.glob_mu(state))  #action batch
            V2 = self.critic.Q_net.Q1_val(state, self.actor.policy_net(state))

        val = torch.exp(V1 - V2)
        val[val > 30] = 30

        loss = torch.sum(
            (self.actor.glob_mu(state) - self.actor.policy_net(state)) ** 2 * alpha * val).mean()

        # loss = torch.sum((self.actor.glob_mu(state) - self.actor.policy_net(state))**2 * alpha * torch.exp(V1 - V2).view(-1,1)).mean()

        # loss = torch.sum(
        #     (self.actor.glob_mu(state) - self.actor.policy_net(state)) ** 2 * alpha * torch.where((V1 - V2)>0, 1, 0)).mean()
        return loss

    def actor_distill(self, actor, Q_net, state, partial):
        loss1 = -Q_net.Q1_val(state, self.actor.policy_net(state)).mean()
        loss2 = torch.sum(
            (actor.glob_mu(state) - actor.policy_net(state)) ** 2).mean()

        distil_loss = partial * loss1 + (1 - partial) * loss2
        self.actor.actor_optimizer.zero_grad()
        distil_loss.backward()

        self.actor.actor_optimizer.step()



    def localDelayUpdate(self, state, Q_net, tau, client_pipe):
        """
        :param state:  state batch from UpdateQ()
        :param Q_net:  critic.Qnet
        :return: 
        """
        self.count += 1
        self.actor.update_policy(state, Q_net)

        if self.count % self.L == 0:
            #
            # if self.count > self.L:
            #     print(temp.item())

            models = [self.actor.policy_net, self.actor.target_net, self.critic.Q_target, self.critic.Q_net]
            self.to_cpu(models)
            client_pipe.send((None, self.actor.policy_net.state_dict(), True))
            mu_params, q_params = client_pipe.recv()  #q_params is half the network
            self.actor.glob_mu.cpu()
            self.actor.glob_mu.load_state_dict(mu_params)
            self.actor.glob_mu.to(self.device)

            self.glob_q.cpu()
            # self.glob_q.client_update(q_params)
            self.glob_q.server_update(self.critic.Q_net, q_params)
            self.glob_q.to(self.device)

            # self.actor.policy_net.load_state_dict(mu_params) #local mu = mu agg

            ####### local distill #######
            if self.actor_dual:
            #     self.to_gpu([self.actor.policy_net, Q_net, self.actor.target_net, self.critic.Q_target])
            #     # self.actor.policy_net.to(self.device)
            #     # partial = 0.1  # default 0.9
            #     for epc in range(self.actor_epc):
            #         state_batch, _, _, _, _ = self.memory.sample(
            #             self.batch_size)
            #         state_batch = torch.tensor(
            #             state_batch, device=self.device, dtype=torch.float)  # bc * state_dim
            #         self.actor_distill(self.actor, Q_net, state_batch, self.actor_partial)
            # #############################
            #     self.actor.update_target(tau, self.actor.policy_net.state_dict())
            #     # self.critic.update_target(tau, q_params) #old
            #     self.critic.update_target(tau, self.critic.Q_net.state_dict())

                return
            else:
                self.actor.policy_net.load_state_dict(mu_params)

                self.actor.update_target(tau, mu_params)
                # self.critic.update_target(tau, q_params) #old
                self.critic.partial_update(tau, q_params)
                self.to_gpu(models)

                return

        #normal update (if distil, not go through)
        self.actor.update_target(tau, self.actor.policy_net.state_dict())
        self.critic.update_target(tau, self.critic.Q_net.state_dict())

    def sync(self, q_params, mu_params):
        self.critic.Q_net.load_state_dict(q_params)
        self.critic.Q_net.to(self.device)
        self.glob_q.load_state_dict(q_params)
        # self.prev_q.load_state_dict(q_params)
        self.glob_q.to(self.device)
        self.critic.Q_target.load_state_dict(q_params)
        self.critic.Q_target.to(self.device)

        self.actor.policy_net.load_state_dict(mu_params)
        self.actor.policy_net.to(self.device)
        # self.actor.glob_mu = mu_params
        self.actor.glob_mu.load_state_dict(mu_params)
        # self.actor.prev_mu.load_state_dict(mu_params)
        self.actor.glob_mu.to(self.device)
        self.actor.target_net.load_state_dict(mu_params)
        self.actor.target_net.to(self.device)

        # self.to_gpu([self.actor.temp_mu, self.actor.prev_mu])

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

