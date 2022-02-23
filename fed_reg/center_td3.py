# from models.Network import mlp_policy, mlp_value
from models.Network import mlp_policy, distill_qnet as mlp_value
from utils.Memory import replay_buffer
import torch.optim as optim
from torch import nn
import torch
import numpy as np

class Actor():
    def __init__(self, state_dim, action_dim, action_bound, args):
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.device = args.device
        self.std_noise = args.action_bound * args.std_noise #std of the noise, when explore
        self.std_policy_noise = args.policy_noise     #std of the noise, when update critics
        self.noise_clip = args.noise_clip
        self.policy_net = mlp_policy(state_dim, action_dim, action_bound).to(args.device)
        self.target_net = mlp_policy(state_dim, action_dim, action_bound).to(args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)

    def predict(self, state):  # for visualize and test
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action = self.policy_net(state).numpy()

        return action

    def choose_action(self, state):
        # for exploration
        # state: 1 * state_dim
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
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
        actor_loss = -Q_net.Q1_val(state, self.policy_net(state)).mean()
        # print(f'actor loss{actor_loss:.2f}')
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


    def update_target(self, tau):
        for params in self.policy_net.state_dict().keys():
            self.target_net.state_dict()[params].copy_(tau * self.policy_net.state_dict()[params] + (1 - tau) * self.target_net.state_dict()[params])

    def load(self, PATH):
        self.policy_net.load_state_dict(torch.load(PATH + 'actor_td3.pth'))
        self.policy_net.cpu()

    def save(self, PATH):
        torch.save(self.policy_net.state_dict(), PATH + "actor_td3.pth")

class Critic():
    def __init__(self,action_dim, state_dim, args):
        self.Q_net = mlp_value(state_dim, action_dim).to(args.device)
        self.Q_target = mlp_value(state_dim, action_dim).to(args.device)
        self.Q_target.load_state_dict(self.Q_net.state_dict())

        self.critic_optimizer = optim.Adam(self.Q_net.parameters(), lr=args.lr)

    def predict(self, state, action):
        q_val1, q_val2 = self.Q_net(state, action)
        return q_val1, q_val2

    def target(self, state, action):
        q_val1, q_val2 = self.Q_target(state, action)
        return q_val1, q_val2

    def update_critics(self):
        pass

    def update_target(self, tau):
        for params in self.Q_net.state_dict().keys():
            self.Q_target.state_dict()[params].copy_(tau * self.Q_net.state_dict()[params] + (1 - tau) * self.Q_target.state_dict()[params])

class TD3():
    def __init__(self, state_dim, action_dim, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = args.device
        self.C_iter = args.C_iter
        self.iter = 0
        self.memory = replay_buffer(args.capacity)
        self.batch_size = args.local_bc
        self.actor = Actor(state_dim, action_dim, args.action_bound, args)
        self.critic = Critic(state_dim, action_dim, args)
        # self.actor_loss = Critic.Q1_net.forward()
        self.critics_loss = nn.MSELoss()

    def UpdateQ(self):
        if len(self.memory) < self.batch_size:
            return
        self.iter += 1
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

        with torch.no_grad():
            # action_tilde = self.actor.choose_action2(state_batch)
            action_tilde = self.actor.choose_action2(n_state_batch)
            q1_target, q2_target = self.critic.target(n_state_batch, action_tilde)

            max_target_q_val = torch.cat((q1_target, q2_target), dim=1).min(1)[0].detach().view(-1, 1)
            y_hat = reward_batch + self.gamma * max_target_q_val * (1 - done_batch)

        current_q_val = self.critic.predict(state_batch, action_batch)

        loss = self.critics_loss(current_q_val[0], y_hat) + self.critics_loss(current_q_val[1], y_hat)
        # print(f'critic loss{loss:.2f}')
        self.critic.critic_optimizer.zero_grad()
        loss.backward()
        self.critic.critic_optimizer.step()

        if self.iter % self.C_iter == 0:
            self.DelayUpdate(state_batch, self.critic.Q_net, self.tau)


    def DelayUpdate(self, state, Q_net, tau):
        """
        :param state:  state batch from UpdateQ()
        :param Q_net:  critic.Qnet
        :return: 
        """
        self.actor.update_policy(state, Q_net)
        self.actor.update_target(tau)
        self.critic.update_target(tau)

        # for param, target_param in zip(Q_net.parameters(), self.critic.Q_target.parameters()):
        #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # for param, target_param in zip(self.actor.policy_net.parameters(), self.actor.target_net.parameters()):
        #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def choose_action(self, state):
        action = self.actor.choose_action(state)
        return action

    def predict(self, state): # for eval
        with torch.no_grad():
            # state = torch.tensor(state, dtype=torch.float).cuda()
            state = torch.tensor(state, dtype=torch.float)
            action = self.actor.policy_net(state).cpu().numpy()

        return action

    def save(self, PATH):
        torch.save(self.actor.policy_net.state_dict(), PATH + "actor_td3.pth")
        torch.save(self.critic.Q_net.state_dict(), PATH + "critic_td3.pth")


    def load(self, PATH):
        self.actor.policy_net.load_state_dict(torch.load(PATH + 'actor_td3.pth'))
        self.actor.target_net.load_state_dict(self.actor.policy_net.state_dict())
        self.critic.Q_net.load_state_dict(torch.load(PATH + 'critic_td3.pth'))
        self.critic.Q_target.load_state_dict(self.critic.Q_net.state_dict())

