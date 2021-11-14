# RLlab
lab for reinforcement learning projects

## FRL

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





