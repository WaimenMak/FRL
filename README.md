# RLlab
lab for reinforcement learning projects

## FRL

### Problem Statement

FL aims to train a global model without sharing the data from the local devices. The major challenges in FL are the training data in the local database are usually non-IID, and the communcations between server and clients are limited to the bandwidth and network environment. 

Although FL has been widely studied in supervised learning tasks, there are still little exploration on how to exploit the FL ideas to deal with the control or decision-making task in dynamic environments. In this work we consider leveraging the advantage of FL to train an robust and general agent that can perform in different environments while protecting the privacy.

### Motivation

Our target is to train the agent locally, with the help of the server, we hope the agent could perform well in similar environments but with different dynamics. The straightforward way to achieve this goal is to collect all the data to train the agent centrally, but to protect the privacy of the agents, we intuitively combine the FL methods with the off-policy RL algorithm TD3. And to better address the aforementioned objective heterogeneity problem, lots of work proposed different kind of local regularizer to constraint the training of the local model. However, these kinds of methods may not be works well in the reinforcement learning setting. As long as the value network trained on the local data, its estimation of different environments would be unreliable and such that leading to a bad performance of the actor network. Besides, naively aggregating the models would be very hard for the RL based algorithm to converge in the early stage of training, resulting in a high communication cost issue. Inspired by the work in \cite{luo2021no} and \cite{lin2020ensemble}, we introduce partial network distillation techniques to aggregate the local models while protecting the data collected by the local agents. And we compare our method with the baseline methods in FL.

### Objective Function

Please refer this link for the inference of the objective function:

[Formula](Formula.ipynb)

### Pseudo code for FedTD3

```python
syncronise local update:
sync Q μ from server
set n = 0;
for t in T:
	agent explore based on μ
	agent saves collected data to local data base
	agent update Q based on target network
	if t mod N == 0:
		agent send Q, wait for server aggregate
		agent receive Qt from server.
	if t mod M == 0:
        n += 1
		agent update μ based on local Q
        if n mod L == 0:
			agent send μ to server, wait for server aggregate.
			agent receive μ from server
		agent update policy target based on current policy net
        agent update Q target based on current Q net
```

### Project Structure

```shell
├─main
│  └─dist_reg.py
│  └─dist_reg_v2.py
│  └─dist_td3.py
│  └─fedregtd3.py
│  └─fedtd3_reg.py
│  └─fed_scaffold.py
│  └─scaffold_td3.py
│  └─central_classic.py
│  └─central_box2d.py
├─models
│  └─Network.py
├─non_stationary_envs
│  └─ContinuousCart.py
│  └─Pendulum.py
│  └─walker.py
├─utils
│  └─Memory.py
│  └─Tools.py
```

The above structure is the main structure of this project. The key programs are contained in the `main` folder. The file start with `dist*` are our method and the others are baseline methods. The file start with `central*` are the methods that collect all the local data and do the training in the server side. Basically it violate the federated setting but it provides information that how far the federated algorithm can achieve.

The network structure used in this project can be seen in `Network.py` in models. The codes in folder `non_stationary_envs` are for generating the local environments. The folder `utils` stores some toolkit for the programs, e.g. the file `Memory.py` implement the replay buffer by utilizing the python data structure `deque` .

#### main

The codes in `main`contain the following basic components: 

```python
def ClientUpdate(client_pipe, agent, local_env, args):
```

This function implement the training of the local agents. It would be run in a multiprocess way, each agent has its own process. Every round of communication it would send the parameters of the critic or actor to the main process.

```python
def ServerUpdate(pipe_dict, server, weighted, actor, envs, args): 
```

This function is for the update of the server. It collects all the models from the local side and do the model fusion. The model fusion function is as follow:

```python
def Agg_q(local_models, global_net, weighted, args):
    """
    :param local_models:  tuple local q_net output layer
    :param global_net:   tuple too
    :param weighted: 
    :param args: 
    :return: 
    """
    with torch.no_grad():
        K = args.client_num
        for i in range(2):
            for params in global_net[i].keys():
                global_net[i][params].copy_(weighted[0] * local_models[0][i][params])
            for params in global_net[i].keys():
                for k in range(1, K):
                    global_net[i][params] += weighted[k] * local_models[k][i][params]
```

In the baseline method, the local models are naively weighted aggregated. In our method, we would do the model distillation based on the pseudo data after aggregating the  models. The following function is the `distillation` part (refer to `dist_reg_v2.py`).

```python
def distill(self, args):
        for epoch in range(args.epochs):
            for data in self.train_loader:
                rep1, rep2, label1, label2 = data
                rep1, rep2, label1, label2 = rep1.to(args.device), rep2.to(args.device), label1.to(args.device), label2.to(args.device)

                oupt1, oupt2 = self.q.server_oupt(rep1, rep2)
                # oupt = torch.cat(oupt)

                self.optimizer.zero_grad()
                loss = self.loss_fn(oupt1, label1) + self.loss_fn(oupt2, label2)
                loss.backward()
                self.optimizer.step()
```

The above `train_loader` is the dataset for the distillation on the representation of the data, each round the server collect the statistic of the representation of the local data, and generate the pseudo data according to the Gaussian distribution which can protect the privacy as much as possible. The following code in `dist_reg_v2.py` are for generating the pseudo data.

```python
dist_rep1, dist_rep2 = agent.critic.Q_net.client_rep(state_batch, action_batch)
agent.to_cpu([agent.critic.Q_net])
# dist_rep1 = torch.tensor(dist_rep1.cpu())
dist_rep1 = dist_rep1.cpu().numpy()
dist_rep2 = dist_rep2.cpu().numpy()

mean1 = np.mean(dist_rep1, axis=0)
mean2 = np.mean(dist_rep2, axis=0)

cov1 = np.cov(dist_rep1.T)
cov2 = np.cov(dist_rep2.T)
dist_rep1 = torch.from_numpy(
    multivariate_normal(mean1, cov1, args.local_bc)).to(torch.float32)
dist_rep2 = torch.from_numpy(
    multivariate_normal(mean2, cov2, args.local_bc)).to(torch.float32)
```



#### TD3 implementation:

The core code of the implementation of Twin Delayed DDPG is in the file `dist_td3.py` and `fedregtd3.py`. These file contain the `actor class` and the `critic class` which are the key component of the TD3 framework.

For different baseline methods, it can be directly change the objective function in `update_policy()` and `localDelayUpdate()` to add different kinds of regular term. The function in `fedregtd3.py` is as follow:

```python
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
```

`mu` and `beta` are the hyper parameters of the baseline method  `Moon` and `FedProx`, and `l2_norm` and `l_con_mu` are the regular term.

### Dataset

As mentioned above,  we generate different local environment with different transition function $p(s_{t+1}|s_t)$.  To achieve this goal,  we randomly change the physical environment parameters in the source code of gym. For example, in the environment `cartpole`, we change the pole length, pole mass and cart mass also the gravity. In the environment `BipedalWalkerHardcore`,   there are three kinds of obstacle. We change the occurence probability of each obstacle to induce the environment heterogeneity. 

![ens](README.assets/ens.jpg)

Another method to simulate heterogeneous environments is to add noise to the input of the agents. In each local environment we add Gaussian noise with different mean to the agent's input. i.e.:
$$
<s,a> + N(\mu_i, \sigma)
$$

