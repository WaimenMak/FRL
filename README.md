# RLlab
lab for reinforcement learning projects

## FRL

### Motivation

Our target is to train the agent locally, with the help of the server, we hope the agent could perform well in similar environments but with different dynamics. The straightforward way to achieve this goal is to collect all the data to train the agent centrally, but to protect the privacy of the agents, we intuitively combine the FL methods with the off-policy RL algorithm TD3. And to better address the aforementioned obective heterogeneity problem, we introduce several regular term to the loss function of TD3 and compare the performance of each method.

### Objective Function

The global objective function is as follow:


$$
\min {F(w)} = \sum_{k=1}^Np_kF_k(w)
$$
In the Q learning setting, our goal is to find the optimal estimation function `Q_*(s,a)` ,   where
$$
Q_*(s,a) = \mathbb{E}_{S_{t+1}\sim p(.|S_t,a_t)}[R_t + \gamma max_a Q_*(S_{t+1},a)|S_t=s_t,A_t=a_t]
$$
we use TD to estimate the expectation,  we hope that we could find $w_*$   such that
$$
\mathbb{E}_{(s_t,s_{t+1})\sim D}[(Q_*(s_t,a_t) - (R_t + \gamma max_a Q_*(s_{t+1},a)))^2] = 0
$$
so the objective function can be:
$$
F(w) = \mathbb{E}_{S\sim D}[(Q_w(s_t,a_t) - (R_t + \gamma max_a Q_w(s_{t+1},a)))^2]
$$
rewrite it:
$$
F(w) = 1/N\sum_j^Nl(w;x_j) \\
l(\omega, x_j) = [Q_\omega(s_t^j,a_t^j) - (r_t^j + \gamma max_a Q_\omega(s_{t+1}^j,a))]^2 \\
x_j : (s_t^j,a_t^j,r_t^j,s_{t+1}^j) \sim D
$$
above is the centralize objective function, i.e. we only have the global agent, and the data $x_j$ is collected from the joint distribution `D`:
$$
P(x) = P(s_t,a_t,r_t,s_{t+1}) = P(s_t)P_w(a_t|s_t)P(s_{t+1}|s_t,a_t)
$$
therefore the data collected from exploration is affected by two distrbution $P(s)$ and $P_w(a|s)$, specifically it is affected by the environment and the current policy of the agent.

with above,  we denote the local objective function of the device `k`as:
$$
F_k(w_k) = 1/n_k\sum_j^{n_k}l(w_k;x_j^k) \\
x_j^k : (s_t^j,a_t^j,r_t^j,s_{t+1}^j) \sim D_k
$$
In this case, if  the environment of device k is different or the policy of the agent k isdifferent, the distribution $D_k$ would be different, and the collected data would be non iid.

The update of the parameters $w_k$:
$$
w_{t+i+1}^k = w_{t+i}^k - \eta\nabla F_k(w_{t+i}^k;x_{t+i}^k), i = 0,1,..,E \\
\nabla F_k(w_{t+i}^k;x_{t+i}^k) = 1/n_k\sum_j^{n_k}\nabla l(w_t^k;x_j^k)
$$
$w_t$ is the parameters loaded from server at time step $t$ .  $E$ is the update frequency.  The update of `FedAvg` in global:
$$
w_{t+E} = \sum_{k =1}^N p_kw_{t+E}^k
$$
$p_k=n_k/n$ , We  assume $n_0 = n_1... = n_k$  and the environment is stationary i.e. $P(s)$  of different device is identity, to prove that when E=1, above update is equivalent to centralized mode update:
$$
w_{t+1} = \sum_{k =1}^N p_kw_{t+1}^k\\
= \sum_{k =1}^N p_k[w_t^k - \eta\nabla F_k(w_t^k;x_t^k)]\\
=\sum_{k=1}^N \frac{n_k}{n}w_t^k - \eta\sum_{k=1}^N \frac{n_k}{n}\frac{1}{n_k}\sum_{j=1}^{n_k}\nabla l(w_t^k;x_{tj}^k) \\
$$
when $n_0 = n_1... = n_k$ and $w_t^k = w_t$:
$$
\sum_{k=1}^N \frac{n_k}{n}w_t^k - \eta\sum_{k=1}^N \frac{n_k}{n}\frac{1}{n_k}\sum_{j=1}^{n_k}\nabla l(w_t^k;x_{tj}^k) \\
=w_t - \eta F(w_t; x_t)
$$
where $x_t=\left\{x_t^k\right\}k\in N$ .

consider that in distributed dqn, there are two ways to collect the data, one is Gorila dqn, randomly choose data from local replay buffer which contains the data collected based on old policy, so the data is from vary distribution. The other is asynchrounous one step dqn, compute the gradient based on current policy. In fedavg, when E = 1, and if the environment is identity, the collected data would be from the identity distribution, but however when E > 1, each local data collected from the device are based on different local policy, so the local data within the device would be very relavant, but data between the device would be very different, this could be affect the averaging. So I think we sill need the replay buffer to randomly collect data.

## Summary

### Topic: Federated Reinforcement Learning for non-stationary environment



### 目前已完成工作：

1. Federated Averaging在CartPole， Pendulum上代码（非并行）

2. 采用TD3算法在BipedalWalker实验

3. fedavg TD3 多进程下通讯的伪代码

4. BipedalWalker可在环境增加障碍物，可以调节障碍物的类型以及出现频率制造non-stationary环境。

   3中的伪代码：

   ```python
   syncronise local update:
   sync Q μ from server
   set n = 0;
   for t in T:
   	agent explore base on μ
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
   
   T is the total iteration, T is the local update times of Q net, T/M is the delay update of policy net in TD3 algorithm, total communication round = T/N + T/(M*L). It can be proved that when N = 1, L = 1 this is equal to centralize TD3. M * L  better be the integer multiple of N, so that the Q target update could be consistent.

### 计划

1. 12/24 前完成多进程版本TD3，比较multi agent和single agent区别，进行fedavg实验
2. 解决identical environment下多个agent用fedavg算法的缺陷，（随着本地更新次数增加模型融合后效果理论上会变差）
3. 解决2中问题后，处理non-stationary环境内训练的agent，模型融合后对server模型污染的问题。

### 2中的一些解决方案

关于fedavg算法用于更新的一些问题：

由于RL中的训练数据是根据agent自身的policy进行收集（即agent网络参数），因此agent在多次本地更新后，不同agent之间的policy会大不同，即使是identical环境，收集到的数据分布也会不一样，因为这里是fedrated的设定，所以不能够通过共享数据来解决这种data distribution shift的问题
$$
\tau = <s_1,a_1,r_1,s_2...s_n> \\
P(\tau) = \prod_{t} P_\omega(a_t|s_t)P_\theta(s_{t+1}|s_t) \\
$$
$P(\tau)$ 的分布取决于参数 $\omega$ 和 $\theta$ ，其中$\omega$是agent policy, $\theta$ 是环境状态转移参数，non stationary环境下参数会不一样。The first term would cause system heterogeneity, second term cause object heterogeneity.

目前想到一个比较简单的解决identical环境下的方法，是通过在训练过程中增加一个惩罚项（KL divergence or L2 norm），以减少local parameter与global parameter之间的差距。

### L2 norm

$$
Q_{loss} = \mathbb{E}_{(s_t,s_{t+1})\sim P(\cdot|\omega;\theta)}[(Q(s_t,a_t) - (R_t + \gamma max_a Q(s_{t+1},a)))^2] + \beta dist(\phi, glob \phi) \\

\mu_{loss} = -Q(s, \mu_w(s)) + \beta dist(\omega, glob \omega)
$$





