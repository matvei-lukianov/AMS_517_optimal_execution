# Recent advances in reinforcement learning in finance

Ben Hambly1 Renyuan Xu2 Huining Yang1
1Mathematical Institute, University of Oxford, Oxford, UK
2Epstein Department of Industrial and Systems Engineering, University of Southern California, Los Angeles, California, USA

## Abstract
The rapid changes in the finance industry due to the
increasing amount of data have revolutionized the tech-
niques on data processing and data analysis and brought
new theoretical and computational challenges. In con-
trast to classical stochastic control theory and other ana-
lytical approaches for solving financial decision-making
problems that heavily reply on model assumptions, new
developments from reinforcement learning (RL) are able
to make full use of the large amount of financial data
with fewer model assumptions and to improve deci-
sions in complex financial environments. This survey
paper aims to review the recent developments and use
of RL approaches in finance. We give an introduction
to Markov decision processes, which is the setting for
many of the commonly used RL approaches. Various
algorithms are then introduced with a focus on value-
and policy-based methods that do not require any model
assumptions. Connections are made with neural net-
works to extend the framework to encompass deep RL
algorithms. We then discuss in detail the application
of these RL algorithms in a variety of decision-making
problems in finance, including optimal execution, port-
folio optimization, option pricing and hedging, market
making, smart order routing, and robo-advising. Our
survey concludes by pointing out a few possible future
directions for research.
## 1 Introduction
The mathematical approach to many financial decision-making problems has traditionally been
through modeling with stochastic processes and using techniques from stochastic control. The
choice of models is often dictated by the need to balance tractability with applicability. Simple
models lead to tractable and implementable strategies in closed-form or that can be found through
traditional numerical methods. However, these models sometimes oversimplify the mechanisms
and the behavior of financial markets which may result in strategies that are suboptimal in prac-
tice and that can potentially result in financial losses. On the other hand, models that try to capture
realistic features of financial markets are much more complex and are often mathematically and
computationally intractable using the classical tools of stochastic optimal control.
In recent years, the availability of large amounts of financial data on transactions, quotes, and
order flows in electronic order-driven markets has revolutionized data processing and statisti-
cal modeling techniques in finance and brought new theoretical and computational challenges
Dixon et al. (2020). In contrast to the classical stochastic control approach, new ideas coming
from reinforcement learning (RL) are being developed to make use of all this information. RL
describes methods by which agents acting within some system might learn to make optimal deci-
sions through repeated experience gained by interacting with the system. In the finance industry,
there have been a number of recent successes in applying RL algorithms in areas such as order exe-
cution, market making, and portfolio optimization that have attracted a lot of attention. This has
led to rapid progress in adapting RL techniques to improve trading decisions in various financial
markets when participants have limited information on the market and other competitors.
Although there are already a number of more specialized review papers concerning aspects of
RL in finance, we aim to review a broad spectrum of activity in this area. This survey is intended
to provide a systematic introduction to the theory of RL, provide a unified framework for per-
formance evaluation and a comprehensive summary of the cutting-edge results in RL theory. It
is followed by an introductory discussion of each of the following financial problems—optimal
execution, portfolio optimization, option pricing and hedging, market making, smart order rout-
ing (SOR), and robo-advising. Moreover, we will also discuss the advantages of RL methods over
classical approaches such as stochastic control, especially for the problems that have been studied
extensively in the mathematical finance literature. For other surveys in the literature on RL appli-
cations in finance, see Charpentier et al. (2021), Fischer (2018), Kolm and Ritter (2020), Meng and
Khushi (2019), and Mosavi et al. (2020). The main focus of these surveys is traditional financial
applications such as portfolio optimization and optimal hedging (Charpentier et al., 2021; Kolm
& Ritter, 2020), or trading on the stock and foreign exchange markets (Meng & Khushi, 2019), or
specific RL approaches such as actor–critic-based methods (Fischer, 2018) and deep RL methods
(Mosavi et al., 2020).
Our survey will begin by discussing Markov decision processes (MDP), the framework for many
RL ideas in finance. We will then consider different approaches to learning within this frame-
work with the main focus being on value- and policy-based methods. In order to implement
these approaches, we will introduce deep reinforcement methods, which incorporate deep learn-

ing ideas in this context. For financial applications, we will consider a range of topics and for
each we will introduce the basic underlying models before considering the RL approach to tack-
ling them. We will discuss a range of papers in each application area and give an indication of
their contributions. We conclude with some thoughts about the direction of development of RL
in finance.
## 2 The Basics Of Reinforcement Learning
RL is an approach to understanding and automating goal-directed learning and decision-making.
It leads naturally to algorithms for such problems and is distinguished from other computational
approaches by its emphasis on learning by the individual from direct interaction with their envi-
ronment, without relying on exemplary supervision or complete models of the environment. RL
uses a formal framework defining the interaction between a learning agent and their environment
in terms of states, actions, and rewards. This framework is intended to be a simple way of repre-
senting essential features of the artificial intelligence problem. These features include a sense of
cause and effect, a sense of uncertainty and nondeterminism, and the existence of explicit goals.
The history of RL has two main threads that were pursued independently before intertwining
in modern RL. One thread concerns learning by trial and error and started in the psychology of
animal learning. The other thread concerns the problem of optimal control for an evolving system
and its solution using value functions and dynamic programming. Although this thread did not
involve learning directly, the Bellman equation developed from this line of literature is viewed as
the foundation of many important modern RL algorithms such as 𝑄-learning and Actor–Critic.
Over the last few years, RL, grounded on combining classical theoretical results with deep learn-
ing and the functional approximation paradigm, has proved to be a fruitful approach to many
artificial intelligence tasks from diverse domains. Breakthrough achievements include reaching
human-level performance in such complex tasks as Go (Silver et al., 2017) and multiplayer Star-
Craft II (Vinyals et al., 2019), as well as the development of ChatGPT (Radford et al., 2019). The
generality of the RL framework allows its application in both discrete and continuous spaces to
solve tasks in both real and simulated environments (Lillicrap et al., 2016).
Classical RL research during the last third of the previous century developed an extensive
theoretical core on which modern algorithms are based. Several algorithms, such as Temporal-
Difference Learning and 𝑄-learning, were developed and are able to solve small-scale problems
when either the states of the environment can be enumerated (and stored in memory) or the opti-
mal policy lies in the space of linear or quadratic functions of the state variable. Although these
restrictions are extremely limiting, these foundations of classical RL theory underlie the mod-
ern approaches, which benefit from increased computational power. Combining this framework
with deep learning (Goodfellow et al., 2016) was popularized by the Deep 𝑄-learning algorithm,
introduced in Mnih et al. (2013), which was able to play any of 57 Atari console games without
tweaking the network architecture or algorithm hyperparameters. This novel approach was exten-
sively researched and significantly improved over the following years (Fan et al., 2020; Gu et al.,
2016; Van Hasselt et al., 2016).
Our aim in this section is to introduce the foundations of RL. We start with the setup for MDP
in Section 2.1 with both an infinite time horizon and a finite time horizon, as there are financial
applications of both settings in the literature. In Section 2.2, we then introduce the learning proce-
dure when the reward and transition dynamics of the MDPs are unknown to the decision-maker.
In particular, we introduce various criteria to measure the performance of learning algorithms in
different settings. Then we focus on two major classes of model-free RL algorithms, value-based

methods in Section 2.3 and policy-based methods in Section 2.4, with an infinite-time horizon.
Finally, Section 2.5 contains a general discussion of (model-based and model-free) RL with a
finite-time horizon, model-based RL with an infinite-time horizon, and regularization methods
for RL.
2.1
Setup: Markov decision processes
We first introduce MDPs with infinite time horizon. Many portfolio optimization problems, such
as those arising from pension funds, investigate long-term investment strategies and hence it is
natural to formulate them as a decision-making problem over an infinite time horizon. We then
introduce MDPs with finite time horizon. Problems such as the optimal execution of the purchase
or liquidation of a quantity of asset usually involve trading over a short time horizon and there
may be a penalty at the terminal time if the targeted amount is not completely executed by then.
Finally, we discuss different policy classes that are commonly adopted by the decision-maker.
Infinite time horizon and discounted reward.
Let us start with a discrete time MDP with an infinite time horizon and discounted reward. We
have a Markov process taking values in a state space , where we can influence the evolution
by taking an action from a set . The aim is to optimize the expected discounted return from
the system by choosing a policy, that is a sequence of actions through time. We formulate this
mathematically by defining the value function 𝑉∗for each 𝑠∈to be
𝑉∗(𝑠) = sup
Π
𝑉Π(𝑠) ∶= sup
Π
𝔼Π[∑∞
𝑡=0 𝛾𝑡𝑟(𝑠𝑡, 𝑎𝑡)|𝑠0 = 𝑠],
(1)
subject to
𝑠𝑡+1 ∼𝑃(𝑠𝑡, 𝑎𝑡),
𝑎𝑡∼𝜋𝑡(𝑠𝑡).
(2)
Here and throughout the paper, 𝔼Π denotes the expectation under the policy Π, where the proba-
bility measure ℙdescribes both dynamics and the rewards in the MDP. We will write () for the
probability measures over a space . The state space (, 𝑑) and the action space (, 𝑑) are both
complete separable metric spaces, which includes the case of and being finite, as often seen
in the RL literature; 𝛾∈(0, 1) is a discount factor; 𝑠𝑡∈and 𝑎𝑡∈are the state and the action at
time 𝑡; 𝑃∶× →() is the transition function of the underlying Markov process; the nota-
tion 𝑠𝑡+1 ∼𝑃(𝑠𝑡, 𝑎𝑡) denotes that 𝑠𝑡+1 is sampled from the distribution 𝑃(𝑠𝑡, 𝑎𝑡) ∈(); the reward
𝑟(𝑠, 𝑎) is a random variable in ℝfor each pair (𝑠, 𝑎) ∈× ; and the policy Π = {𝜋𝑡}∞
𝑡=0 is Marko-
vian, in that it just depends on the current state, and can be either deterministic or randomized.
For a deterministic policy, 𝜋𝑡∶→maps the current state 𝑠𝑡to a deterministic action. On the
other hand, a randomized policy 𝜋𝑡∶→() maps the current state 𝑠𝑡to a distribution over
the action space. For a randomized policy 𝜋𝑡, we will use 𝜋𝑡(𝑠) ∈() and 𝜋𝑡(𝑠, 𝑎) to represent,
respectively, the distribution over the action space given state 𝑠and the probability of taking action
𝑎at state 𝑠. In this case, with infinite-time horizon, we assume that the reward 𝑟and the transi-
tion dynamics 𝑃are time homogeneous, which is a standard assumption in the MDP literature
(Puterman, 2014). We also note that the setup is essentially the same if we consider the problem
of minimizing costs rather than maximizing a reward.

The well-known dynamic programming principle (DPP), that the optimal policy can be
obtained by maximizing the reward from one step and then proceeding optimally from the new
state, can be used to derive the following Bellman equation for the value function (1)
𝑉∗(𝑠) = sup
𝑎∈
{
𝔼[𝑟(𝑠, 𝑎)] + 𝛾𝔼𝑠′∼𝑃(𝑠,𝑎)[𝑉∗(𝑠′)]
}
.
(3)
We write the value function as
𝑉∗(𝑠) = sup
𝑎∈
𝑄∗(𝑠, 𝑎),
where the 𝑄-function, one of the basic quantities used for RL, is defined to be
𝑄∗(𝑠, 𝑎) = 𝔼[𝑟(𝑠, 𝑎)] + 𝛾𝔼𝑠′∼𝑃(𝑠,𝑎)[𝑉∗(𝑠′)],
(4)
the expected reward from taking action 𝑎at state 𝑠and then following the optimal policy
thereafter. There is also a Bellman equation for the 𝑄-function given by
𝑄∗(𝑠, 𝑎) = 𝔼[𝑟(𝑠, 𝑎)] + 𝛾𝔼𝑠′∼𝑃(𝑠,𝑎) sup
𝑎′∈
𝑄∗(𝑠′, 𝑎′).
(5)
This allows us to retrieve the optimal (stationary) policy 𝜋∗(𝑠, 𝑎) (if it exists) from 𝑄(𝑠, 𝑎), in that
𝜋∗(𝑠, 𝑎) ∈arg max𝑎∈𝑄(𝑠, 𝑎).
An alternative setting over an infinite time horizon is the case of average reward, which is
also referred to as an ergodic reward. This ergodic setting is not as relevant to financial applica-
tions and hence will not be covered in this review paper. We refer the reader to Abbasi-Yadkori
et al. (2019); Wei et al. (2020) for a more detailed discussion of RL with infinite-time horizon and
average reward.
Finite time horizon.
When there is a finite time horizon 𝑇< ∞, we no longer discount future values and have a
terminal reward. The MDP problem with finite time horizon can be expressed as
𝑉∗
𝑡(𝑠) = sup
Π
𝑉Π
𝑡(𝑠) ∶= sup
Π
𝔼Π
[𝑇−1
∑
𝑢=𝑡
𝑟𝑢(𝑠𝑢, 𝑎𝑢) + 𝑟𝑇(𝑠𝑇)
||||
𝑠𝑡= 𝑠
]
, ∀𝑠∈,
(6)
subject to
𝑠𝑢+1 ∼𝑃𝑢(𝑠𝑢, 𝑎𝑢),
𝑎𝑢∼𝜋𝑢(𝑠𝑢), 𝑡≤𝑢≤𝑇−1.
(7)
As in the infinite horizon case, we denote by 𝑠𝑢∈and 𝑎𝑢∈the state and the action of
the agent at time 𝑢. However, where this differs from the infinite horizon case, is that we allow
time-dependent transition and reward functions. Now 𝑃𝑢∶× →() denotes the transition
function and 𝑟𝑢(𝑠, 𝑎) a real-valued random variable for each pair (𝑠, 𝑎) ∈× for 𝑡≤𝑢≤𝑇−1.
Similarly 𝑟𝑇(𝑠), the terminal reward, is a real-valued random variable for all 𝑠∈. Finally, the
Markovian policy Π = {𝜋𝑢}𝑇
𝑡=0 can be either deterministic or randomized.

The corresponding Bellman equation for the value function (6) is defined as
𝑉∗
𝑡(𝑠) = sup
𝑎∈
{
𝔼[𝑟𝑡(𝑠, 𝑎)] + 𝔼𝑠′∼𝑃𝑡(𝑠,𝑎)[𝑉∗
𝑡+1(𝑠′)]
}
,
(8)
with the terminal condition 𝑉∗
𝑇(𝑠) = 𝔼[𝑟𝑇(𝑠)]. We write the value function as
𝑉∗
𝑡(𝑠) = sup
𝑎∈
𝑄∗
𝑡(𝑠, 𝑎),
where the 𝑄∗
𝑡function is defined to be
𝑄∗
𝑡(𝑠, 𝑎) = 𝔼[𝑟𝑡(𝑠, 𝑎)] + 𝔼𝑠′∼𝑃𝑡(𝑠,𝑎)[𝑉∗
𝑡(𝑠′)].
(9)
There is also a Bellman equation for the 𝑄-function given by
𝑄∗
𝑡(𝑠, 𝑎) = 𝔼[𝑟𝑡(𝑠, 𝑎)] + 𝔼𝑠′∼𝑃𝑡(𝑠,𝑎)
[
sup
𝑎′∈
𝑄∗
𝑡+1(𝑠′, 𝑎′)
]
,
(10)
with the terminal condition 𝑄∗
𝑇(𝑠, 𝑎) = 𝔼[𝑟𝑇(𝑠)] for all 𝑎∈. We note that since financial time
series data are typically nonstationary, time-varying transition kernels and reward functions in
Equations (6) and (7) are particularly important for financial applications.
For an infinite horizon MDP with finite state and action space, and finite reward 𝑟, a further use-
ful observation is that such an MDP always has a stationary optimal policy, whenever an optimal
policy exists.
Theorem 2.1 Theorem 6.2.7 in Puterman (2014). Assume || < ∞, || < ∞, and |𝑟| < ∞with
probability one. For any infinite horizon discounted MDP, there always exists a deterministic
stationary policy that is optimal.
Theorem 2.1 implies that there always exists a fixed policy so that taking actions specified by
that policy at each time step maximizes the discounted reward. The agent does not need to change
policies with time. There is a similar result for the average reward case, see Theorem 8.1.2 in
Puterman (2014). This insight reduces the question of finding the best sequential decision-making
strategy to the question of finding the best stationary policy. To this end, we write 𝜋∶→()
(without a time index) for a stationary policy throughout the paper.
Linear MDPs and linear functional approximation.
Linear MDPs have been extensively studied in the recent literature on theoretical RL in order to
establish tractable and efficient algorithms. In a linear MDP, both the transition kernels and the
reward function are assumed to be linear with respect to some feature mappings (Bradtke & Barto,
1996; Melo & Ribeiro, 2007).
In the infinite horizon setting, an MDP is said to be linear with a feature map 𝜙∶× →ℝ𝑑,
if there exist 𝑑unknown (signed) measures 𝜇= (𝜇(1), … , 𝜇(𝑑)) over and an unknown vector
𝜃∈ℝ𝑑such that for any (𝑠, 𝑎) ∈× , we have
𝑃(⋅|𝑠, 𝑎) = ⟨𝜙(𝑠, 𝑎), 𝜇(⋅) ⟩,
𝑟(𝑠, 𝑎) = ⟨𝜙(𝑠, 𝑎), 𝜃⟩.
(11)

Similarly for the finite horizon setting, an MDP is said to be linear with a feature map 𝜙∶
× →ℝ𝑑, if for any 0 ≤𝑡≤𝑇, there exist 𝑑unknown (signed) measures 𝜇𝑡= (𝜇(1)
𝑡, … , 𝜇(𝑑)
𝑡)
over and an unknown vector 𝜃𝑡∈ℝ𝑑such that for any (𝑠, 𝑎) ∈× , we have
𝑃𝑡(⋅|𝑠, 𝑎) = ⟨𝜙(𝑠, 𝑎), 𝜇𝑡(⋅) ⟩,
𝑟𝑡(𝑠, 𝑎) = ⟨𝜙(𝑠, 𝑎), 𝜃𝑡⟩,
(12)
Typically features are assumed to be known to the agent and bounded, namely, ‖𝜙(𝑠, 𝑎)‖ ≤1 for
all (𝑠, 𝑎) ∈× .
The linear MDP framework is closely related to the literature on RL with linear functional
approximation, where the value function is assumed to be of the form
𝑄(𝑠, 𝑎) = ⟨𝜓(𝑠, 𝑎), 𝜔⟩,
𝑉(𝑠) = ⟨𝜉(𝑠), 𝜂⟩
(13)
for the infinite horizon case, and
𝑄𝑡(𝑠, 𝑎) = ⟨𝜓(𝑠, 𝑎), 𝜔𝑡⟩,
𝑉𝑡(𝑠) = ⟨𝜉(𝑠), 𝜂𝑡⟩, ∀0 ≤𝑡≤𝑇
(14)
for the finite horizon case. Here 𝜓∶× →ℝ𝑑and 𝜉∶→ℝ𝑑are known feature mappings
and 𝜔, 𝜔𝑡𝜂, and 𝜂𝑡are unknown vectors. It has been shown that the linear MDP (i.e., Equation 11
or 12) and the linear functional approximation formulation (i.e., Equation 13 or 14) are equiva-
lent under mild conditions (Jin et al., 2020; Yang & Wang, 2019). In addition to linear functional
approximations, we refer the reader to Dai et al. (2018) for a general discussion of RL with non-
linear functional approximations and to Section 3.2 for neural network approximation, one of the
most popular nonlinear functional approximations used in practice.
Nonlinear functional approximation.
Compared to linear functional approximations, nonlinear functional approximations do not
require knowledge of the kernel functions a priori. Nonlinear functional approximation could
potentially address the mis-specification issue when the agent has an incorrect understanding of
the functional space that the MDP lies in. The most popular nonlinear functional approximation
approach is to use neural networks, leading to deep RL. Thanks to the universal approximation
theorem, this is a theoretically sound approach and neural networks show promising performance
for a wide range of applications. Meanwhile gradient-based algorithms for certain neural net-
work architectures enjoy provable convergence guarantees. We defer the discussion of deep RL to
Section 3.2 and its financial applications to Section 4.
## 2.2 From MDP to learning
When the transition dynamics 𝑃and the reward function 𝑟for the infinite horizon MDP problem
are unknown, this MDP becomes a RL problem, which is to find an optimal stationary policy 𝜋
(if it exists) while simultaneously learning the unknown 𝑃and 𝑟. The learning of 𝑃and 𝑟can
be either explicit or implicit, which leads to model-based and model-free RL, respectively. The
analogous ideas hold for the finite horizon case. We introduce some standard RL terminology.
A more detailed introduction to RL can be found in textbooks such as Sutton and Barto (2018),
Powell (2021).

Agent–environment interface.
In the RL setting, the learner or the decision-maker is called the agent. The physical world that
the agent operates in and interacts with, comprising everything outside the agent, is called the
environment. The agent and environment interact at each of a sequence of discrete time steps,
𝑡= 0, 1, 2, 3, …, in the following way. At the beginning of each time step 𝑡, the agent receives some
representation of the environment’s state, 𝑠𝑡∈and selects an action 𝑎𝑡∈. At the end of this
time step, in part as a consequence of its action, the agent receives a numerical reward 𝑟𝑡(pos-
sibly stochastic) and a new state 𝑠𝑡+1 from the environment. The tuple (𝑠𝑡, 𝑎𝑡, 𝑟𝑡, 𝑠𝑡+1) is called a
sample at time 𝑡and ℎ𝑡∶= {(𝑠𝑢, 𝑎𝑢, 𝑟𝑢, 𝑠𝑢+1)}𝑡
𝑢=0 is referred to as the history or experience up to
time 𝑡. An RL algorithm is a finite sequence of well-defined policies for the agent to interact with
the environment.
Exploration versus exploitation.
In order to maximize the accumulated reward over time, the agent learns to select her actions
based on her past experiences (exploitation) and/or by trying new choices (exploration). Explo-
ration provides opportunities to improve performance from the current suboptimal solution to
the ultimate globally optimal one, yet it is time consuming and computationally expensive as
over-exploration may impair the convergence to the optimal solution. Meanwhile, pure exploita-
tion, that is, myopically picking the current solution based solely on past experience, though
easy to implement, tends to yield suboptimal global solutions. Therefore, an appropriate trade-off
between exploration and exploitation is crucial in the design of RL algorithms in order to improve
the learning and the optimization performance. See Section 2.5.3 for a more detailed discussion.
Simulator.
In the procedure described above on how the agent interacts with the environment, the agent
does so in an online fashion. Namely, the initial state at time step 𝑡+ 1, 𝑠𝑡+1, is the state that the
agent moves to after taking action 𝑎𝑡from state 𝑠𝑡. This is a challenging setting since an efficient
exploration scheme is needed. For instance, 𝑄-learning with the 𝜀-greedy policy (to be introduced
in Section 2.3.3) may take exponentially many episodes to learn the optimal policy (Kearns &
Singh, 2002). Some results assume access to a simulator (Koenig ... Simmons, 1993) (a.k.a., a
generative model (Azar et al., 2012)), which allows the algorithm to query arbitrary state-action
pairs and return the reward and the next state. This implies that the agent can “restart” the system
at any time. That is, the initial state at time step 𝑡+ 1 does not need to be the state that agent
moves to after taking action 𝑎𝑡from state 𝑠𝑡. The “simulator” significantly alleviates the difficulty
of exploration, since a naive exploration strategy which queries all state-action pairs uniformly at
random already leads to the most efficient algorithm for finding an optimal policy (Azar et al.,
2012).
Randomized policy versus deterministic policy.
A randomized policy 𝜋𝑡∶→() is also known in the control literature as a relaxed con-
trol and in game theory as a mixed strategy. Despite the existence of a deterministic optimal
policy as stated in Theorem 2.1 for the infinite time horizon case, most of the RL algorithms
adopt randomized policies to encourage exploration when RL agents are not certain about
the environment.

An example: multi-armed bandits.
We illustrate these points by discussing multi-armed bandit problems, a special case of RL prob-
lems. The multi-armed bandit is a model for a set of slot machines. A simple version is that there
are a number of arms, each with a stochastic reward coming from a fixed probability distribution,
which is initially unknown. We try these arms in some order, which may depend on the sequence
of rewards that have been observed so far and attempt to maximize our return. A common objec-
tive in this context is to find a policy for choosing the next arm to be tried, under which the sum of
the expected rewards comes as close as possible to the ideal reward, that is, the expected reward
that would be obtained if we were to try the “best” arm at all times. Thus, the agent interacts with
the environment by selecting an action—pulling an arm—and receiving a reward. The policy of
the order of pulling on the arms has to balance exploitation, the continued use of an arm that is
returning a decent reward, with exploration, sampling arms about which there is less information
to find one with a possibly better return.
In this setting for a multi-armed bandit problem, we have not involved state dynamics. In this
case, an admissible policy is simply a distribution over the action space for a randomized policy
and a single arm for a deterministic policy. A simple example of a bandit with a state space is
the stochastic contextual bandit, which is used extensively in the personalized advertisement rec-
ommendation and clinical treatment recommendation literature. In this case, the state dynamics
(also referred to as the context) are sampled i.i.d. from an unknown distribution. For the per-
sonalized advertisement recommendation example, the state at time 𝑡can be interpreted as the
personal information and purchasing behavior of a targeted client.
The problem was first considered in the seminal work of Robbins (1952), which derives poli-
cies that asymptotically attain an average reward that converges in the limit to the reward of the
best arm. The multi-armed bandit problem was later studied in discounted, Bayesian, Markovian,
expected reward, and adversarial setups. See Berry and Fristedt (1985) for a review of the classical
results on the multi-armed bandit problem.
## 2.2.1 Performance evaluation
It is not possible to solve MDPs analytically in general and therefore we will discuss numerical
algorithms to determine the optimal policies. In order to assess different algorithms, we will need
a measure of their performance. Here we will consider several types of performance measure:
sample complexity, rate of convergence, regret analysis, and asymptotic convergence.
For RL with a finite time horizon, one episode contains a sequence of states, actions, and
rewards, which starts at time 0 and ends at the terminal time 𝑇, and the performance is eval-
uated in terms of the total number of samples in all episodes.1 Sometimes, we also refer to this
setting as episodic RL. For RL with infinite time horizon, the performance is evaluated in terms
of the number of steps. In this setting, one step contains one (state, action, reward, next state)
tuple. It is worth noting that the episodic criterion can also be used to analyze infinite horizon
problems. One popular technique is to truncate the trajectory (with infinite steps) at a large time
and hence translate the problem into an approximating finite time horizon problem. Examples
include REINFORCE (Zhang et al., 2021) and the policy gradient method (Liu et al., 2020) (see
Section 2.4).
1 Sample complexity with respect to the number of episodes has also been adopted in the literature (Dann & Brunskill,
2015; Dann et al., 2017). The translation between these two criteria is straightforward.

Throughout the analysis, we use the notation ˜𝑂(⋅) to hide logarithmic factors when describing
the order of the scaling in 𝜀, 𝛿, ||, and || (when these are finite), and other possible model
parameters such as the discount rate 𝛾in the infinite horizon case or the dimension 𝑑of the fea-
tures when functional approximation is used. We write ˜′ rather than ˜to indicate that, although
there may be other parameters, we only include the dependence on 𝜀. We denote by poly(⋅) a
polynomial function of its arguments.
Sample complexity.
In RL, a sample (𝑠, 𝑎, 𝑟, 𝑠′) is defined as a tuple consisting of a state 𝑠, an action 𝑎, an instantaneous
reward received by taking action 𝑎at state 𝑠, and the next state 𝑠′ observed afterwards. Sample
complexity is defined as the total number of samples required to find an approximately optimal
policy. This evaluation criterion can be used for any kind of RL problem.
For episodic RL, an algorithm returns, after the end of the (𝑘−1)th episode with 𝑀𝑘−1 sam-
ples used in this episode, a policy 𝜋𝑘to be applied in the 𝑘th episode. The sample complexity of
this algorithm is the minimum number ∑𝐾
𝑘=1 𝑀𝑘−1(𝜀) of samples such that for all 𝑘≥𝐾, 𝜋𝑘is
𝜀-optimal with probability at least 1 −𝛿, that is, for 𝑘≥𝐾,
ℙ
(
𝑉𝜋𝑘
0
≥𝑉∗
0 −𝜀
)
≥1 −𝛿.
(15)
For discounted RL, an algorithm returns, after the end of the (𝑛−1)th step with 𝑀𝑛−1 samples
used in this iteration, a policy 𝜋𝑛to be applied in the 𝑛th step. There are several notions of sample
complexity in this setting. The most commonly used ones are defined through the value function
and the 𝑄-function with all input variables (i.e., 𝑠∈for 𝑉and (𝑠, 𝑎) ∈× for 𝑄) and they are
referred to as the 𝑉-sample complexity and 𝑄-sample complexity in the literature. The 𝑄-sample
complexity of a given algorithm is defined as the minimum number ∑𝑁
𝑛=1 𝑀𝑛−1(𝜀) of samples
such that for all 𝑛≥𝑁, 𝑄(𝑛) is 𝜀-close to 𝑄∗, that is
‖𝑄(𝑛) −𝑄∗‖∞∶=
sup
𝑠∈,𝑎∈
|||𝑄∗(𝑠, 𝑎) −𝑄(𝑛)(𝑠, 𝑎)||| ≤𝜀,
(16)
holds either with high probability (that is ℙ(‖𝑄(𝑛) −𝑄∗‖∞< 𝜀) ≥1 −𝛿) or in the expectation
sense 𝔼[‖𝑄(𝑛) −𝑄∗‖∞] < 𝜀. Similarly, the 𝑉-sample complexity is defined as the minimum
number ∑𝑁
𝑛=1 𝑀𝑛−1(𝜀) of samples such that for all 𝑛≥𝑁, 𝑉(𝑛) is 𝜀-close to 𝑉∗, that is
‖𝑉(𝑛) −𝑉∗‖∞∶= sup
𝑠∈
|||𝑉(𝑛)(𝑠) −𝑉∗(𝑠)||| ≤𝜀,
(17)
holds either with high probability (in that ℙ(‖𝑉(𝑛) −𝑉∗‖∞< 𝜀) ≥1 −𝛿) or in the expecta-
tion sense 𝔼[‖𝑉(𝑛) −𝑉∗‖∞] < 𝜀. Note that Equation (16) implies Equation (17) since 𝑉(𝑛)(𝑠) =
sup𝑎∈𝑄(𝑛)(𝑠, 𝑎) and 𝑉∗(𝑠) = sup𝑎∈𝑄∗(𝑠, 𝑎). In our analysis, we do not distinguish between
whether the sample complexity bounds for Equations (16) and (17) are in the high probability
sense or in the expectation sense.
The second type of sample complexity, the sample complexity of exploration, is defined as the
number of samples such that the nonstationary policy 𝜋𝑛at time 𝑛is not 𝜀-optimal for the current
state 𝑠𝑛. This measure counts the number of mistakes along the whole trajectory. Mathematically
speaking, the sample complexity of exploration for a given algorithm is the minimum number
∑𝑁
𝑛=1 𝑀𝑛−1(𝜀) of samples such that for all 𝑛≥𝑁, 𝜋𝑛is 𝜀-optimal when starting from the current

state 𝑠𝑛with probability at least 1 −𝛿, that is, for 𝑛≥𝑁,
ℙ(|𝑉𝜋𝑛(𝑠𝑛) −𝑉∗(𝑠𝑛)| ≤𝜀) ≥1 −𝛿.
(18)
Note that the condition ℙ(‖𝑉(𝑛) −𝑉∗‖∞< 𝜀) ≥1 −𝛿associated with Equation (17) is stronger
and implies the condition in Equation (18).
Another weaker criterion, the mean-square sample complexity, measures the sample complexity
in the average sense with respect to the steady state distribution or the initial distribution 𝜇. In
this case, the mean-square sample complexity is defined as the minimum number ∑𝑁
𝑛=1 𝑀𝑛−1(𝜀)
of samples such that for all 𝑛≥𝑁,
‖𝑉𝜋𝑛−𝑉∗‖𝜇∶=
√
∫
(𝑉𝜋𝑛(𝑠) −𝑉∗(𝑠))2𝜇(𝑑𝑠) ≤𝜀.
(19)
Since Equation (19) is an aggregated guarantee via the 𝑉function over a state distribution 𝜇, it is
implied by Equation (17).
Rate of convergence.
To emphasize the convergence with respect to the number of iterations required, we introduce the
notion of rate of convergence, which uses the relationship between the number of iterations/steps
𝑁and the error term 𝜀, to quantify how fast the learned policy converges to the optimal solution.
This is also called the iteration complexity in the RL and optimization literature. Compared to the
notion of sample complexity introduced above, the rate of convergence calculates the number of
iterations needed to reach a given accuracy while ignoring the number of samples used within
each iteration. When only one sample is used in each iteration, the rate of convergence coincides
with the sample complexity. In addition when a constant number of samples (i.e., independent
from 𝜀) are used within each iteration, the rate of convergence and the sample complexity are of
the same order with respect to 𝜀.
In particular, an algorithm is said to converge at a linear rate if 𝑁∼˜′(log(1∕𝜀)). Similarly, an
algorithm is said to converge at a sublinear rate (slower than linear) if 𝑁∼˜′(1∕𝜀𝑝) with 𝑝≥1,
and is said to converge at a superlinear (faster than linear) if 𝑁∼˜′(log(log(1∕𝜀)).
Regret analysis.
The regret of a given policy 𝜋is defined as the difference between the cumulative reward of the
optimal policy and that gathered by 𝜋. It quantifies the exploration–exploitation trade-off.
For episodic RL with horizon 𝑇in formulation (6), the regret of an algorithm after 𝐾episodes
with 𝑀= 𝑇𝐾steps is
𝑅(𝑀) =
‖‖‖‖‖‖
𝐾× 𝑉∗
0 −
𝐾
∑
𝑘=1
𝔼[𝑉𝜋𝑘
0 ]
‖‖‖‖‖‖∞
= sup
𝑠∈
(
𝐾× 𝑉∗
0 −
𝐾
∑
𝑘=1
𝔼[𝑉𝜋𝑘
0 ]
)
.
(20)
There is currently no regret analysis for RL problems with infinite time horizon and discounted
reward. The natural definition of regret as given above is less meaningful in this case as the
cumulative discounted reward is bounded and hence does not scale with 𝑇.

Asymptotic convergence.
In addition to sample complexity and regret analysis discussed above, asymptotic convergence is
another weaker convergence guarantee, which requires the error term to decay to zero as 𝑁goes
to infinity (without specifying the order of 𝑁). This is often a first step in the theoretical analysis
of RL algorithms.
## 2.2.2 Classification of RL algorithms
An RL algorithm includes one or more of the following components:
∙a representation of a value function that provides a prediction of how good each state or each
state/action pair is,
∙a direct representation of the policy 𝜋(𝑠) or 𝜋(𝑠, 𝑎),
∙a model of the environment (the estimated transition function and the estimated reward func-
tion) in conjunction with a planning algorithm (any computational process that uses a model
to create or improve a policy).
The first two components are related to what is called model-free RL. When the latter component
is used, the algorithm is referred to as model-based RL.
In the MDP setting, model-based algorithms maintain an approximate MDP model by estimat-
ing the transition probabilities and the reward function, and derive a value function from the
approximate MDP. The policy is then derived from the value function. Examples include Braf-
man and Tennenholtz (2002), Kearns and Singh (2002), Lattimore and Hutter (2012) and Szita and
Szepesvári (2010). Another line of model-based algorithms make structural assumptions on the
model, using some prior knowledge, and utilize the structural information for algorithm design.
For example, see Fazel et al. (2018) for learning linear-quadratic regulators (LQRs) over an infinite
time horizon and Hambly et al. (2021), Basei et al. (2021) for the finite time horizon case.
Unlike the model-based method, model-free algorithms directly learn a value (or state-value)
function or the optimal policy without inferring the model. Model-free algorithms can be further
divided into two categories, value-based methods and policy-based methods. Policy-based meth-
ods explicitly build a representation of a policy and keep it in memory during learning. Examples
include policy gradient methods and trust-region policy optimization (TRPO) methods. As an
alternative, value-based methods store only a value function without an explicit policy during the
learning process. In this case, the policy is implicit and can be derived directly from the value
function (by picking the action with the best value).
In our discussion of methodology, we focus on model-free RL algorithms for MDP with infinite
horizon and discounted reward. In particular, we introduce some classical value- and policy-based
methods in Sections 2.3 and 2.4, respectively. For the episodic setting and model-based algorithms,
see the discussion in Section 2.5.
## 2.3 Value-based methods
In this section, we introduce the methodologies of several classical value-based methods in the
setting of an infinite time horizon with discounting, finite state and action spaces (|| < ∞and
|| < ∞), and stationary policies. The setting with finite state and action spaces is also referred to
as the tabular case in the RL literature. For more general setups with an infinite number of actions

A L G O R I T H M 1 TD(0) Method for estimating 𝑉𝜋
and states, we refer the reader to Section 2.4.4 for a discussion of linear functional approximations
and to Section 3.2 for neural network approximations.
Recall that for RL, the distribution of the reward function 𝑟and the transition function 𝑃is
unknown to the agent. Therefore, the expectations in the Bellman equations (3) and (5) cannot be
calculated directly. On the other hand, the agents can observe samples (𝑠, 𝑎, 𝑟, 𝑠′) by interacting
with the system without a model of the system’s dynamics 𝑃or any prior knowledge of 𝑟. The
agents can then learn the value function or 𝑄-function using the samples. Following this idea, we
next introduce the classical temporal-difference learning algorithm, with 𝑄-learning and State-
Action-Reward-State-Action (SARSA) as two special cases.
## 2.3.1 Temporal-difference learning
Given some samples (𝑠, 𝑎, 𝑟, 𝑠′) obtained by following a policy 𝜋, the agent can update her estimate
of the value function 𝑉𝜋(defined in Equation 1) at the (𝑛+ 1)th iteration by
𝑉𝜋,(𝑛+1)(𝑠) ←(1 −𝛽𝑛(𝑠, 𝑎))
𝑉𝜋,(𝑛)(𝑠)
⏟⎴⏟⎴⏟
current estimate
+𝛽𝑛(𝑠, 𝑎) [𝑟+ 𝛾𝑉𝜋,(𝑛)(𝑠′)]
⏟⎴⎴⎴⎴⏟⎴⎴⎴⎴⏟
new estimate
,
(21)
with some initialization of the value function 𝑉𝜋,(0). Here, 𝛽𝑛(𝑠, 𝑎) is the learning rate at itera-
tion 𝑛+ 1, which balances the weighting between the current estimate and the new estimate.
The quantity 𝛽𝑛can be a constant, can depend on the current state 𝑠, or even depend on the
observed samples up to iteration 𝑛+ 1. Algorithm 1 provides some pseudo-code for an imple-
mentation of the TD method. Equation (21) is sometimes referred to as a TD(0) method. This is
a special case of a TD(𝜆) method (for some 𝜆∈[0, 1]) which is a combination of TD learning
(with weight 𝜆) and a Monte Carlo method (with weight 1 −𝜆). Here a Monte Carlo method indi-
cates a simulation-based method to calculate the value function with a longer period of data (in
the episode-by-episode sense) rather than a single sample in each update. See Section 2.4.2 for
a detailed discussion of the REINFORCE method, which is an example of such a Monte Carlo
method. Equation (21) is also referred to as a one-step TD method as it only uses information
from one update step of the system. There are multistep TD methods; see Chapter 7 and Chapter
12 in Sutton and Barto (2018) for more details.

A L G O R I T H M 2 𝑄-learning with samples from a policy 𝜋
The TD update in Equation (21) can also be written as
𝑉𝜋,(𝑛+1)(𝑠) ←𝑉𝜋,(𝑛)(𝑠) + 𝛽𝑛(𝑠, 𝑎)
⎡
⎢
⎢
⎢⎣
TD target
⏞⎴⎴⎴⏞⎴⎴⎴⏞
𝑟+ 𝛾𝑉𝜋,(𝑛)(𝑠′) −𝑉𝜋,(𝑛)(𝑠)
⎤
⎥
⎥
⎥⎦
⏟⎴⎴⎴⎴⎴⎴⎴⎴⏟⎴⎴⎴⎴⎴⎴⎴⎴⏟
TD error 𝛿𝑛
.
(22)
The term 𝛿𝑛, commonly called the TD error, measures the difference between the estimated value
at 𝑠and the better estimate 𝑟+ 𝛾𝑉𝜋,(𝑛)(𝑠′), which is often called the TD target in the literature.
The TD error and TD target are two important components in analyzing the convergence of the
approximate value function and arise in various forms in the RL literature.
2.3.2
𝑄-learning algorithm
A version of TD learning is 𝑄-learning. This is a stochastic approximation to the Bellman equa-
tion (5) for the 𝑄-function with samples observed by the agent. At iteration 𝑛(1 ≤𝑛≤𝑁−1), the
𝑄-function is updated using a sample (𝑠, 𝑎, 𝑟, 𝑠′) where 𝑠′ ∼𝑃(𝑠, 𝑎),
𝑄(𝑛+1)(𝑠, 𝑎) ←(1 −𝛽𝑛(𝑠, 𝑎))
𝑄(𝑛)(𝑠, 𝑎)
⏟⎴⏟⎴⏟
current estimate
+𝛽𝑛(𝑠, 𝑎)
[
𝑟(𝑠, 𝑎) + 𝛾max
𝑎′ 𝑄(𝑛)(𝑠′, 𝑎′)
]
⏟⎴⎴⎴⎴⎴⎴⎴⎴⏟⎴⎴⎴⎴⎴⎴⎴⎴⏟
new estimate
.
(23)
Here 𝛽𝑛(𝑠, 𝑎) is the learning rate, which balances the weight between the current estimate 𝑄(𝑛)
from iteration 𝑛and the new estimate 𝑟(𝑠, 𝑎) + 𝛾max𝑎′ 𝑄(𝑛)(𝑠′, 𝑎′) calculated using the sample
(𝑠, 𝑎, 𝑟, 𝑠′). Algorithm 2 provides pseudo-code for an implementation of Q-learning.
The following theorem guarantees the asymptotic convergence of the 𝑄-learning update (23)
to the 𝑄-function defined in Equation (4).
Theorem 2.2 Watkins and Dayan (1992). Assume || < ∞and || < ∞. Let 𝑛𝑖(𝑠, 𝑎) be the index
of the 𝑖th time that the action 𝑎is used in state 𝑠. Let 𝑅< ∞be a constant. Given bounded rewards
|𝑟| ≤𝑅, learning rate 0 ≤𝛽𝑛< 1 and
∞
∑
𝑖=1
𝛽𝑛𝑖(𝑠,𝑎) = ∞,
∞
∑
𝑖=1
(𝛽𝑛𝑖(𝑠,𝑎))2 < ∞, ∀𝑠, 𝑎,
(24)

A L G O R I T H M 3 SARSA: On-policy TD learning
then 𝑄(𝑛)(𝑠, 𝑎) →𝑄⋆(𝑠, 𝑎) as 𝑛→∞, ∀𝑠, 𝑎with probability 1.
This is a classical result (one of the first few theoretical results in RL) and the proof of the
convergence is based on the action–replay process (ARP), which is an artificial controlled Markov
process constructed from the episode sequence and the learning rate sequence 𝛽. Theorem 2.2
implies that eventually 𝑄(𝑛) will convergence to the true value function 𝑄⋆when condition (24)
is satisfied. However, this asymptotic result does not provide any insights on how many samples
are needed to reach a given accuracy. More results on the sample complexity for 𝑄-learning-related
algorithms are discussed in Section 2.3.4.
## 2.3.3 Sarsa
In contrast to the 𝑄-learning algorithm, which takes samples from any policy 𝜋as the input where
these samples could be collected in advance before performing the 𝑄-learning algorithm, SARSA
adopts a policy, which is based on the agent’s current estimate of the 𝑄-function. The different
source of samples is indeed the key difference between on-policy and off-policy learning, which
will be discussed after the SARSA algorithm.
The policy derived from 𝑄(𝑛) using 𝜀-greedy (lines 4 and 8 in Algorithm 3) suggests the following
choice of action 𝑎when in state 𝑠:
{
select from arg max𝑎′ 𝑄(𝑛)(𝑠, 𝑎′)
with probability 1 −𝜀,
uniformly sample from 
with probability 𝜀.
(26)
In Algorithm 3, SARSA uses the 𝑄(𝑛) function with an 𝜖-greedy policy to choose 𝑎′ and to per-
form exploration. In contrast, Q-learning uses the maximum value of 𝑄(𝑛) over all possible actions
for the next step, which is equivalent to setting 𝜀= 0 when choosing 𝑎′.
In addition to the 𝜀-greedy policy, other exploration methods such as the softmax operator
(resulting in a Boltzmann policy) (Asadi & Littman, 2017; Haarnoja et al., 2017; Wang et al., 2020)
can also be applied. See Section 2.5.3 for a more detailed discussion.
On-policy learning versus off-policy learning.
An off-policy agent learns the value of the optimal policy independently of the agent’s actions. An
on-policy agent learns the value of the policy being carried out by the agent including the explo-

ration steps. 𝑄-learning is an off-policy agent as the samples (𝑠, 𝑎, 𝑟, 𝑠′) used in updating Equation
(23) may be collected from any policy and may be independent of the agent’s current estimate of
the 𝑄-function. The convergence of the 𝑄-function relies on the exploration scheme of the pol-
icy 𝜋(see Theorem 2.2). In contrast to 𝑄-learning, SARSA is an on-policy learning algorithm and
uses only its own estimation of the 𝑄-function to select an action and receive a real-time sample
in each iteration. The difference is seen in how the new estimate in the update is calculated. In
the update step of 𝑄-learning (23), we have max𝑎′ 𝑄(𝑛)(𝑠′, 𝑎′), which is the maximum value of the
𝑄-function at a given state 𝑠′. On the other hand, the new estimate in the update of SARSA (25)
takes 𝑄(𝑛)(𝑠′, 𝑎′) where 𝑎′ is selected based on a policy derived from 𝑄(𝑛) (via the 𝜀-greedy policy
(26)).
## 2.3.4 Discussion
In this section, we provide a brief overview of sample complexity results for model-free and value-
based RL with infinite-time horizon and discounted reward. Sample complexity results for the
model-based counterpart are deferred to Section 2.5.
There has been very little nonasymptotic analysis of TD(0) learning. Existing results have
focused on linear function approximations. For example, Dalal et al. (2018) showed that the mean-
squared error converges at a rate of ˜′(
1
𝜀1∕𝜎) with decaying learning rate 𝛽𝑛=
1
(1+𝑛)𝜎for some
𝜎∈(0, 1) where i.i.d. samples are drawn from a stationary distribution; Lakshminarayanan and
Szepesvari (2018) provided an improved ˜′( 1
𝜀) bound for iterate-averaged TD(0) with constant
step-size; and Bhandari et al. (2018) reached the same mean-square sample complexity ˜′( 1
𝜀) with
a simpler proof and extends the framework to the case where non-i.i.d. but Markov samples are
drawn from the online trajectory. A similar analysis was applied by Zou et al. (2019) to SARSA and
𝑄-learning algorithms for a continuous state space using linear function approximation (see the
end of Section 2.1).
The 𝑄-sample complexity for the 𝑄-learning algorithm of ˜(
||||
(1−𝛾)5𝜀5∕2 poly(log 𝛿−1)) was first
established in Even-Dar et al. (2003) with decaying learning rate 𝛽𝑛=
1
(1+𝑛)4∕5 and access to a
simulator. This was followed by Beck and Srikant (2012), which showed that the same order of 𝑄-
sample complexity ˜(
||||
(1−𝛾)5𝜀5∕2 ) could be achieved with a constant learning rate 𝛽𝑛≡(1−𝛾)4𝜀2
|||| (𝑛≤
𝑁). Without access to a simulator, delayed 𝑄-learning Strehl et al. (2009) was shown to achieve a
sample complexity of exploration of order ˜( ||||
(1−𝛾)8𝜀4 poly(log 𝛿−1)) assuming a known reward. An
improvement to this result to ˜( ||||
(1−𝛾)7𝜀4 poly(log 𝛿−1)) has recently been obtained using an upper-
confidence-bound (UCB) exploration policy (Wang et al., 2020) and a single trajectory. In addition,
sample complexities of 𝑄-learning variants with (linear) function approximations have been
established in several recent papers. Yang and Wang (2019) assumed a linear MDP framework (see
the end of Section 2.1) and the authors provided a V-sample complexity ˜(
𝑑
(1−𝛾)3𝜀2 poly(log 𝛿−1))
with 𝑑denoting the dimension of the features. This result matches the information-theoretic
lower bound up to log(⋅) factors. Later work, Lattimore et al. (2020) considered the setting where
the transition kernel can be approximated by linear functions with small approximation errors
to make the model more robust to model mis-specification. In this case, the authors provided a
V-sample complexity of order ˜(
𝐾
(1−𝛾)4𝜀2 poly(log 𝛿−1)). Although the dependence on 𝛾is not as

good as in Yang and Wang (2019), this framework can be applied to a more general class of models
which are “near” linear. Although 𝑄-learning is one of the most successful algorithms for finding
the best action-value function 𝑄, its implementation often suffers from large high biased esti-
mates of the 𝑄-function values incurred by random sampling. The double 𝑄-learning algorithm
proposed in Hasselt (2010) tackled this high estimation issue by randomly switching the update
between two 𝑄-estimators, and has thus gained significant popularity in practice. The first sam-
ple complexity result for the double 𝑄-learning algorithm was established in Xiong et al. (2020)
with a Q-sample complexity on the order of ˜((
1
(1−𝛾)6𝜀2 ln
||||
(1−𝛾)7𝜀2𝛿)1∕𝜔+ ( 1
1−𝛾ln
1
(1−𝛾)2𝜀)1∕(1−𝜔))
and learning rate 𝛽𝑛=
1
𝑛𝑤for some constant 𝜔∈(0, 1).
## 2.4 Policy-based methods
So far we have focused on value-based methods where we learn the value function to generate
the policy using, for instance, the 𝜀-greedy approach. However, these methods may suffer from
high computational complexity as the dimensionality of the state or action space increases (for
example, in continuous or unbounded spaces). In this case, it may be more effective to directly
parametrize the policy rather than the value function. Furthermore, it is empirically observed that
policy-based methods have better convergence properties than value-based methods (Dabérius
et al., 2019; Sewak, 2019; Yu & Sun, 2020). This stems from the fact that value-based methods can
have big oscillations during the training process since the choice of actions may need to change
dramatically in order to have an arbitrarily small change in the estimated value functions.
In this section, we focus on model-free policy-based methods, where we learn a parametrized
policy without inferring the value function in the setting of infinite time horizon with discounting,
finite state and action spaces, and stationary policies. Examples of policy-based methods include
REINFORCE (Williams, 1992), Actor–Critic methods (Konda ... Tsitsiklis, 2000), TRPO (Schul-
man et al., 2015), proximal policy optimization (PPO) (Schulman et al., 2017) among others. We
first parametrize the policy 𝜋by a vector 𝜃, thus we define 𝜋(𝑠, ⋅; 𝜃) ∈(), the probability dis-
tribution parameterized by 𝜃over the action space given the state 𝑠at time 𝑡. Here we will use
𝜋(𝑠, 𝑎; 𝜃) and 𝜋𝜃interchangeably to represent the policy parameterized by 𝜃. We write 𝜇𝜋𝜃for the
stationary distribution of the state under policy 𝜋(𝑠, 𝑎; 𝜃). Then the policy objective function is
defined to be
𝐽(𝜃) ∶= ∫
𝑉𝜋𝜃(𝑠)𝜇𝜋𝜃(𝑑𝑠)
for RL with infinite time horizon, or
𝐽(𝜃) ∶= ∫
𝑉𝜋𝜃
0 (𝑠)𝜇𝜋𝜃(𝑑𝑠)
for RL with finite time horizon. In order to maximize this function, the policy parameter 𝜃is
updated using the gradient ascent rule:
𝜃′ = 𝜃+ 𝛽ˆ
∇𝜃𝐽(𝜃),
(27)
where 𝛽is the learning rate, and ˆ
∇𝜃𝐽(𝜃) is an estimate of the gradient of 𝐽(𝜃) with respect to 𝜃.

A L G O R I T H M 4 REINFORCE: Monte Carlo policy gradient
## 2.4.1 Policy gradient theorem
Our task now is to estimate the gradient ∇𝜃𝐽(𝜃). The objective function 𝐽(𝜃) depends on both the
policy and the corresponding stationary distribution 𝜇𝜋𝜃, and the parameter 𝜃affects both of them.
Calculating the gradient of the action with respect to 𝜃is straightforward given the parametriza-
tion 𝜋(𝑠, 𝑎; 𝜃), whereas it might be challenging to analyze the effect of 𝜃on the state when the
system is unknown. Fortunately, the well-known policy gradient theorem provides an expression
for ∇𝜃𝐽(𝜃), which does not involve the derivatives of the state distribution with respect to 𝜃. We
write the 𝑄-function under policy 𝜋(𝑠, 𝑎; 𝜃) as 𝑄𝜋𝜃(𝑠, 𝑎), then we have the following theorem (for
more details see Sutton ... Barto, 2018).
Theorem 2.3 Policy Gradient Theorem (Sutton et al., 2000). Assume that 𝜋(𝑠, 𝑎; 𝜃) is differentiable
with respect to 𝜃and that there exists 𝜇𝜋𝜃, the stationary distribution of the dynamics under policy
𝜋𝜃, which is independent of the initial state 𝑠0. Then the policy gradient is
∇𝜃𝐽(𝜃) = 𝔼𝑠∼𝜇𝜋𝜃,𝑎∼𝜋𝜃[∇𝜃ln 𝜋(𝑠, 𝑎; 𝜃)𝑄𝜋𝜃(𝑠, 𝑎)].
(28)
For MDPs with finite state and action space, the stationary distribution exists under mild
assumptions. For MDPs with infinite state and action space, more assumptions are needed, for
example, uniform geometric ergodicity (Konda, 2002).
2.4.2
REINFORCE: Monte Carlo policy gradient
Based on the policy gradient expression in Equation (28), it is natural to estimate the expectation
by averaging over samples of actual rewards, which is essentially the spirit of a standard Monte
Carlo method that repeatedly simulates a trajectory of length 𝑀within each iteration. Now we
introduce our first policy-based algorithm, called REINFORCE (Williams, 1992) or Monte Carlo
policy gradient (see Algorithm 4). We use a simple empirical return function 𝐺𝑡to approximate
the 𝑄-function in Equation (28). The return 𝐺𝑡is defined as the sum of the discounted rewards
and given by 𝐺𝑡∶= ∑𝑀
𝑠=𝑡+1 𝛾𝑠−𝑡−1𝑟𝑠. Note that REINFORCE is a Monte Carlo algorithm since
it employs the complete sample trajectories, that is, when estimating the return from time 𝑡, it
uses all future rewards up until the end of the trajectory (see line 7 in Algorithm 4). Within the
𝑛-iteration the policy parameter 𝜃(𝑛+1) is updated 𝑀times with 𝐺𝑡for 𝑡= 0, 1, … , 𝑀−1.

A L G O R I T H M 5 Actor–Critic algorithm
As a standard Monte Carlo method, REINFORCE provides an unbiased estimate of the policy
gradient; however, it suffers from high variance, which, therefore, may lead to slow convergence.
A popular variant of REINFORCE is to subtract a baseline, which is a function of the state 𝑠,
from the return 𝐺𝑡. This reduces the variance of the policy gradient estimate while keeping the
mean of the estimate unchanged. A commonly used baseline is an estimate of the value function,
ˆ𝑉(𝑠), which can be learnt using one of the methods introduced in Section 2.3. Replacing the 𝐺𝑡in
Equation (29) by 𝐺𝑡−ˆ𝑉(𝑠) gives a REINFORCE algorithm with baseline.
2.4.3
Actor–Critic methods
The fact that REINFORCE provides unbiased policy estimates but may suffer from high variance
is an example of the bias-variance dilemma (see, e.g., François-Lavet et al. (2019) and Von Luxburg
and Schölkopf (2011)), which occurs in many machine learning problems. Also, as a Monte Carlo
algorithm, REINFORCE requires complete trajectories so we need to wait until the end of each
episode to collect the data. Actor–Critic methods can resolve the above issues by learning both
the value function and the policy. The learned value function can improve the policy update
through, for example, introducing bias into the estimation. Actor–Critic methods can also learn
from incomplete experience in an online manner.
In Actor–Critic methods the value function is parametrized by 𝑤and the policy is parametrized
by 𝜃. These methods consist of two models: a critic which updates the value function or 𝑄-function
parameter 𝑤, and an actor, which updates the policy parameter 𝜃based on the information given
by the critic. A natural idea is to use policy-based methods for the actor and use one of the methods
introduced in Section 2.3 for the critic; for example SARSA or 𝑄-learning. We give the pseu-
docode for a simple Actor–Critic method in Algorithm 5. Here for the critic, we approximate the
𝑄-function by a linear combination of features, that is, 𝑄(𝑠, 𝑎; 𝑤) = 𝜙(𝑠, 𝑎)⊤𝑤for some known
feature 𝜙∶× →ℝ𝑑, and use TD(0) to update the parameter 𝑤to minimize the least-square
error of the TD error with the gradient of the least-square error taking the form 𝛿𝜙(𝑠, 𝑎) (see lines
9 and 10 in Algorithm 5). For the actor, we use the (vanilla) policy-based method to update the
policy parameter 𝜃(see line 11 in Algorithm 5).
There are three main ways to execute the algorithm. In the nested-loop setting (see, e.g., Kumar
et al., 2019; Xu et al., 2020a), the actor updates the policy in the outer loop after the critic’s repeated
updates in the inner loop. The second way is the two time-scale setting (see, e.g., Xu et al., 2020b),
where the actor and the critic update their parameters simultaneously with different learning
rates. Note that the learning rate for the actor is typically much smaller than that of the critic in this

setting. In the third way, the single-scale setting, the actor and the critic update their parameters
simultaneously but with a much larger learning rate for the actor than for the critic (see, e.g., Fu
et al., 2021; Xu et al., 2021).
## 2.4.4 Discussion
Variants of policy-based methods.
There have been many variations of policy-based methods with improvements in various direc-
tions. For example, the natural policy-based algorithms (Kakade, 2001; Peters ... Schaal, 2008)
modify the search direction of the vanilla version by involving a Fisher information matrix in
the gradient ascent updating rule, which speeds the convergence significantly. To enhance the
stability of learning, we may request the updated policy estimate to be not too far away from
the previous estimate. Along this line, TRPO (Schulman et al., 2015) uses the Kullback–Leibler
(KL)-divergence to measure the change between the consecutive updates as a constraint, while
PPO (Schulman et al., 2017) incorporate an adaptive KL-penalty or a clipped objective in the objec-
tive function to eliminate the constraint. When using the KL-penalty, PPO can be viewed as a
Lagrangian relaxation of the TRPO algorithm. To reduce the sample complexity, deterministic
policy gradient (DPG) (Silver et al., 2014) uses a deterministic function (rather than the stochas-
tic policy 𝜋) to represent the policy to avoid sampling over actions (although enough exploration
needs to be guaranteed in other ways); Actor–Critic with Experience Replay (ACER) (Wang et al.,
2017) reuses some experience (tuples of data collected in previous iterations/episodes, see the dis-
cussion of Deep 𝑄-Networks in Section 3.2), which improves the sample efficiency and decreases
the data correlation. In addition, we mention Asynchronous Advantage Actor–Critic (A3C) and
Advantage Actor–Critic (A2C), two popular Actor–Critic methods with a special focus on parallel
training Mnih et al. (2016). The latter is a synchronous, deterministic version of the former. For
continuous-time framework, see developments in Jia and Zhou (2021, 2022).
Convergence guarantees.
Now we discuss the convergence guarantees of policy-based algorithms. Sutton et al. (2000)
provided the first asymptotic convergence result for policy gradient methods with arbitrary dif-
ferentiable function approximation for MDPs with bounded reward. They showed that such
algorithms, including REINFORCE and Actor–Critic methods, converge to a locally stationary
point. For more examples of asymptotic convergence, see, for example, Konda and Tsitsiklis (2000)
and Bhatnagar (2010).
Based on recent progress in nonconvex optimization, nonasymptotic analysis of policy-based
methods were first established for convergence to a stationary point. For example, Kumar et al.
(2019) provided a convergence rate analysis for a nested-loop Actor–Critic algorithm to the
stationary point through quantifying the smallest number of actor updates 𝑘required to attain
inf 0≤𝑚≤𝑘‖∇𝐽(𝜃(𝑘))‖2 < 𝜀. We denote this smallest number as 𝐾. When the actor uses a policy
gradient method, the critic achieves 𝐾≤˜′(1∕𝜀4) by employing TD(0), 𝐾≤˜′(1∕𝜀3) by employ-
ing the gradient temporal difference, and 𝐾≤˜′(1∕𝜀5∕2) by employing the accelerated gradient
temporal difference, with continuous state and action spaces. Zhang et al. (2020) investigated
a policy gradient method with Monte Carlo estimation of the policy gradient, called the RPG
algorithm, where the rollout length of trajectories is drawn from a Geometric distribution. This
generates unbiased estimates of the policy gradient with bounded variance, which was shown
to converge to the first order stationary point with diminishing or constant learning rate at a
sublinear convergence rate. They also proposed a periodically enlarged learning rate scheme and

proved that the modified RPG algorithm with this scheme converges to the second order station-
ary point in a polynomial number of steps. For more examples of sample complexity analysis for
convergence to a stationary point, see, for example, Papini et al. (2018), Xu et al. (2020a, 2020b),
Shen et al. (2019), Xiong et al. (2021). The global optimality of stationary points was studied
in Bhandari and Russo (2019) where they identified certain situations under which the policy
gradient objective function has no suboptimal stationary points despite being nonconvex.
Recently, there have been results on the sample complexity analysis of the convergence to
the global optimum. Cen et al. (2020) proved that the entropy regularized natural policy gradi-
ent method achieves a 𝑄-sample complexity (and 𝜀-optimal policies) with linear convergence
rate. Shani et al. (2020) showed that with high probability at least 1 −𝛿, the TRPO algorithm
has sublinear rate ˜′(1∕𝜀2) with mean-squared sample complexity ˜( ||2(||+log(1∕𝛿))
(1−𝛾)3𝜀4
) in the
unregularized (tabular) MDPs, and has an improved rate ˜′(1∕𝜀) with mean-squared sample
complexity ˜( ||2(||+log(1∕𝛿))
(1−𝛾)4𝜀3
) in MDPs with entropy regularization. Xu et al. (2020b) gave the
first nonasymptotic convergence guarantee for the two time-scale natural Actor–Critic algorithms
with mean-squared sample complexity of order ˜(
1
(1−𝛾)9𝜀4 ). For single-scale Actor–Critic meth-
ods, the global convergence with sublinear convergence rate was established in both Fu et al.
(2021) and Xu et al. (2021). The nonasymptotic convergence of policy-based algorithms is shown
in other settings, see Zhang et al. (2021) for the regret analysis of the REINFORCE algorithm for
discounted MDPs and Agarwal et al. (2021), Mei et al. (2020) for policy gradient methods in the
setting of known model parameters.
## 2.5 General discussion
So far we have focused on model-free RL with discounting over an infinite time horizon. In this
section, we discuss two other cases: RL over a finite time horizon with both model-based and
model-free methods, and model-based RL with an infinite time horizon.
## 2.5.1 RL with finite time horizon
As discussed earlier, episodic RL with finite time horizon has also been widely used in many
financial applications. In this section, we discuss some episodic RL algorithms (with both model-
based and model-free methods) and their performance through both sample complexity and
regret analysis. Ian et al. (2013) proposed a model-based algorithm, known as posterior sam-
pling for reinforcement learning (PSRL), which is a model-based algorithm, and establishes an
˜(𝑇||
√
||𝑀) expected regret bound. Dann and Brunskill (2015) proposed a so-called upper-
confidence fixed-horizon (UCFH) RL algorithm that is model-based and has 𝑉-sample complexity
of order ˜( ||2|| 𝑇3
𝜀2
) assuming known rewards. Azar et al. (2017) considered two model-based
algorithms in the setting of known reward, called the UCBVI-CH and UCBVI-BF algorithms,
which were proved to achieve a regret bound of ˜(𝑇
√
|| ||𝑀) (when 𝑀≥𝑇||3|| and
|||| ≥𝑇) and ˜(
√
𝑇|| ||𝑀) (when 𝑀≥𝑇3||3|| and |||| ≥𝑇), respectively. Dann
et al. (2017) proposed an upper bounding the expected next state value (UBEV) algorithm, which
achieves a sample complexity3 of ˜( ||2|| 𝑇5
𝜀2
polylog( 1
𝛿)). They also proved that the algorithm has
3 In the original papers, the sample complexity is defined in terms of the number of episodes. Here we multiply the original
order in these papers by 𝑇to match the definition of sample complexity in this paper.

a regret bound of ˜(𝑇2√
|| ||𝑀) with 𝑀≥||5||3. Jin et al. (2018) proved that 𝑄-learning
with UCB exploration achieves regret of ˜(
√
𝑇3|| ||𝑀) without requiring access to a simu-
lator. The above regret bounds depend on the size of the state and action space and thus may
suffer from the curse of dimensionality as || and || increase. To tackle this issue, Yang and
Wang (2020) learned a low-dimensional representation of the probability transition model and
proved that their MatrixRL algorithm achieves a regret bound of ˜(𝑇2 𝑑
√
𝑀), which depends on
the number of features 𝑑rather than || and ||. Jin et al. (2020) provided a ˜(
√
𝑑3𝑇3𝑀) regret
bound with high probability for a modified version of the Least-Squares Value Iteration (LSVI)
algorithm with UCB exploration.
## 2.5.2 Model-based RL with infinite time horizon
To derive policies by utilizing estimated transition probabilities and reward functions under the
infinite time horizon setting, Brafman and Tennenholtz (2002) proposed an R-MAX algorithm,
which can also be applied in zero-sum stochastic games. The R-MAX algorithm was proved to
achieve a sample complexity of exploration of order ˜( ||2||
(1−𝜆)6𝜀3 ) by Li (2009). The Model-based
Interval Estimation (MBIE) in Strehl and Littman (2005) was proved to achieve a sample com-
plexity of exploration of the same order as R-MAX. Szita and Szepesvári (2010) further modified
the R-MAX algorithm to an MoRmax algorithm with an improved sample complexity of explo-
ration of order ˜( || ||
(1−𝜆)6𝜀2 ). Azar et al. (2013) proved that the model-based 𝑄-Value Iteration (QVI)
achieves the 𝑄-sample complexity of ˜( || ||
(1−𝛾)3𝜀2)) for some small 𝜀with high probability at least
1 −𝛿. Agarwal et al. (2020) studied the approximate MDP model with any accurate black-box
planning algorithm and showed that this has the same sample complexity as Azar et al. (2013),
but with a larger range of 𝜀. See also Strehl et al. (2009), Kearns and Singh (2002), Lattimore and
Hutter (2012) for model-based RL along this line.
Linear-quadratic (LQ) problems are another type of model-based RL problem that assumes
linear state dynamics and quadratic objective functions with continuous state and action space.
These problems are one of the few optimal control problems for which there exists a closed-form
analytical representation of the optimal feedback control. Thus LQ frameworks, including the
(single-agent) LQR and (multi-agent) LQ games, can be used in a wide variety of applications,
such as the optimal execution problems which we will discuss later. For a theoretical analysis of
RL algorithms within the LQ framework, see, for example, Fazel et al. (2018) and Hambly et al.
(2021) for the global linear convergence of policy gradient methods in LQR problems with infinite
and finite horizon, respectively. See also Basei et al. (2021) and Guo et al. (2021) for the regret
analysis of LQ RL algorithms in continuous time.
## 2.5.3 Exploration exploitation trade-offs
As mentioned earlier, exploitation versus exploration is a critical topic in RL. RL agents want to
find the optimal solution as fast as possible, meanwhile committing to solutions too fast without
enough exploration could lead to local minima or total failure. Modern RL algorithms that opti-
mize for the best returns can achieve good exploitation quite efficiently, while efficient exploration
remains a challenging and less understood topic.

For multi-armed bandit problems or MDP with finite state and action spaces, classic exploration
algorithms such as 𝜀-greedy, UCB, Boltzmann exploration, and Thompson sampling show promis-
ing performance in different settings. For the 𝜀-greedy algorithm (Dann et al., 2022; Dabney et al.,
2020), the agent randomly explores occasionally with probability 𝜀and takes the optimal action
most of the time with probability 1 −𝜀. For example, see Equation (26) for the 𝜀-greedy method
combined with a value-based algorithm. The UCB type of algorithms are mainly designed for
value-based algorithms (Azar et al., 2017; Jin et al., 2018, 2020; Wang et al., 2020). The agent selects
the greediest action to maximize the UCB 𝑄(𝑠, 𝑎) + 𝑈(𝑠, 𝑎), where 𝑄(𝑠, 𝑎) is the Q value function
and 𝑈(𝑠, 𝑎) is a function inversely proportional to how many times the state–action pair (𝑠, 𝑎) has
been taken (hence proportional to the agent’s confidence in her estimation of the Q function).
For Boltzmann exploration (Asadi & Littman, 2017; Haarnoja et al., 2017; Wang et al., 2020), the
agent draws actions from a Boltzmann distribution (softmax) over the learned Q value function,
regulated by a temperature parameter. For the Thompson sampling method (Ouyang et al., 2017;
Gopalan & Mannor, 2015; Thompson, 1933), the agent keeps track of a belief of the optimal action
via a distribution over the set of admissible actions (for each given state).
For deep RL training when neural networks are used for function approximation, entropy reg-
ularization (adding an entropy term into the loss function) and noise-based exploration (adding
noise into the observation, action, or even parameter space) are often used for better exploration
of the parameter space.
## 2.5.4 Regularization in reinforcement learning
Many recent developments in RL benefit from regularization, which has been shown to improve
not only the exploration but also the robustness. Examples include both policy-based methods
and value-based methods in various settings. Here we provide a general overview of different
regularization techniques and their advantages along with several examples. For example, TRPO
and PPO use the KL-divergence between two consecutive policies as a penalty in policy improve-
ment (Schulman et al., 2015, 2017). Soft-𝑄-learning uses Shannon entropy as a penalty in value
iteration (Haarnoja et al., 2017). The authors confirmed in simulated experiments that entropy reg-
ularization helps to improve exploration and allows transferring skills between tasks. Geist et al.
(2019) proposed a unified framework to analyze the above algorithms via a regularized Bellman
operator. Cen et al. (2020) showed that entropy regularization can also help to speed up the conver-
gence rate from a theoretical perspective. It is worth noting that Wang et al. (2020) explained the
exploitation–exploration trade-off with entropy regularization from a continuous-time stochas-
tic control perspective and provided theoretical support for Gaussian exploration from the special
case of LQR. Recently, Gao et al. (2020) and Tang et al. (2021) extended this idea to a general control
framework beyond the LQ case. Grau-Moya et al. (2018) extended the idea of Shannon’s entropy
regularization to mutual-information regularization and showed its superior performance when
actions have significantly different importance. When functional approximations are applied in
RL training, the regularization technique is shown to be a powerful, flexible, and principled way
to balance approximation and estimation errors (massoud Farahmand et al., 2009; Farahmand
et al., 2008). The idea of regularization is that one starts from a large function space and controls
the solution’s complexity by a regularization (or penalization) term. Examples of regularization
include the Hilbert norm (for 𝑄-functions) and the 𝑙1 norm (for parameters).

## 2.5.5 Reinforcement learning in nonstationary environment
Most existing work on RL considers a stationary environment and aims to find the optimal policy
or a policy with low (static) regret. In many financial applications, however, the environment is
far from being stationary. We introduce the mathematical formulations of nonstationary RL (in
both episodic and infinite horizon settings) below.
Nonstationary episodic RL.
For the episodic (or finite-time horizon) setting, an agent interacts with a nonstationary MDP for
𝐾episodes, with each episode containing 𝑇timestamps. Let (𝑡, 𝑘) denote a (time,episode) index
for the 𝑡th timestamp in the 𝑘th episode. The nonstationary environment can be denoted by a tuple
(, , 𝑇,𝑃,𝑟), where is the state space, is the action space, 𝑇is the number of timestamps
in one episode, 𝑃= {𝑃𝑘,𝑡}1≤𝑘≤𝐾,0≤𝑡≤𝑇is the set of transition kernels with 𝑃𝑘,𝑡∶× →() for
all 1 ≤𝑘≤𝐾and 0 ≤𝑡≤𝑇, and 𝑟= {𝑟𝑘,𝑡}1≤𝑘≤𝐾,0≤𝑡≤𝑇is the set of reward functions with 𝑟𝑘,𝑡(𝑠, 𝑎)
a random variable in ℝ(depending on both the time-step and the episode number) for each pair
(𝑠, 𝑎) ∈× and for 0 ≤𝑡≤𝑇−1. Similarly, the terminal reward 𝑟𝑘,𝑇(𝑠) is a real-valued random
variable for each 𝑠∈. Denote by Π = {𝜋𝑘,𝑡}1≤𝑘≤𝐾,0≤𝑡≤𝑇−1 the Markovian policy in which 𝜋𝑘,𝑡can
be either deterministic or randomized. The value function for this policy at the 𝑘th episode can
be written as
𝑉Π
𝑘,𝑡(𝑠) = 𝔼Π
[𝑇−1
∑
𝑢=𝑡
𝑟𝑘,𝑢(𝑠𝑢, 𝑎𝑢) + 𝑟𝑘,𝑇(𝑠𝑇)
||||||
𝑠𝑡= 𝑠
]
,
(30)
where 𝑠𝑢+1 ∼𝑃𝑘,𝑢(𝑠𝑢, 𝑎𝑢) and 𝑎𝑢∼𝜋𝑘,𝑢(𝑠𝑢).
Nonstationary RL with infinite time horizon.
For the infinite horizon setting, we drop the index indicating episodes and let the finite horizon 𝑇
become infinite. Then the nonstationary environment can be denoted by a tuple (, ,𝑃,𝑟), with
the obvious changes from the episodic case. Again we write Π = {𝜋𝑡}𝑡≥0 for the Markovian policy
in which 𝜋𝑡can be either deterministic or randomized and consider the associated value function
with a discount factor 𝛾:
𝑉Π
𝑡(𝑠) = 𝔼Π
[ ∞
∑
𝑢=𝑡
𝛾𝑢−𝑡𝑟𝑢(𝑠𝑢, 𝑎𝑢)
|||||
𝑠𝑡= 𝑠
]
,
(31)
where 𝑠𝑢+1 ∼𝑃𝑢(⋅|𝑠𝑢, 𝑎𝑢) and 𝑎𝑢∼𝜋𝑢(𝑠𝑢).
Indeed, there has been a recent surge of theoretical studies on this topic for various settings
such as infinite-horizon MDPs with finite state and action spaces (Cheung et al., 2019; Gajane
et al., 2018), episodic MDPs with finite state and action spaces (Mao et al., 2020), and episodic
linear MDPs (Touati ... Vincent, 2020). However, all these papers assume that the agent has prior
knowledge of the degree of nonstationarity such as the number of changes in the environment,
which is not very practical for many applications. To relax this assumption, Cheung et al. (2020)
propose a Bandit-over-Reinforcement-Learning (BoRL) algorithm to extend their earlier result
(Cheung et al., 2019). However, it introduces extra overheads and leads to suboptimal regret. More
recently, Wei and Luo (2021) propose a general approach that is applicable to various RL settings

(including both episodic MDPs and infinite-horizon MDPs) and achieves optimal dynamic regret
without any prior knowledge of the degree of nonstationarity.
## 3 Deep Reinforcement Learning
In the setting where the state and action spaces are very large, or high dimensional or they
are continuous, it is often challenging to apply classical value-based and policy-based methods.
In this context, parametrized value functions (𝑄(𝑠, 𝑎; 𝜃) and 𝑉(𝑠, 𝑎; 𝜃)) or policies (𝜋(𝑠, 𝑎; 𝜃))
are more practical. Here we focus on neural-network-based parametrizations and functional
approximations, which have the following advantages:
∙Neural networks are well suited for dealing with high-dimensional inputs such as time series
data, and, in practice, they do not require an exponential increase in data when adding extra
dimensions to the state or action space.
∙In addition, they can be trained incrementally and make use of additional samples obtained as
learning happens.
In this section, we will introduce deep RL methods in the setting of an infinite time horizon
with discounting, finite state and action spaces (|| < ∞and || < ∞), and stationary policies.
## 3.1 Neural networks
A typical neural network is a nonlinear function, represented by a collection of neurons. These
are typically arranged as a number of layers connected by operators such as filters, poolings, and
gates, that map an input variable in ℝ𝑛to an output variable in ℝ𝑚for some 𝑛, 𝑚∈ℤ+.
In this subsection, we introduce three popular neural network architectures including fully
connected neural networks (FNNs), convolutional neural networks (CNNs) (LeCun ... Bengio,
1995) and recurrent neural networks (Sutskever et al., 2011).
FNNs.
A FNN is the simplest neural network architecture where any given neuron is connected to all
neurons in the previous layer. To describe the setup, we fix the number of layers 𝐿∈ℤ+ in the
neural network and the width of the 𝑙th layer 𝑛𝑙∈ℤ+ for 𝑙= 1, 2, … , 𝐿. Then for an input variable
𝐳∈ℝ𝑛, the functional form of the FNN is
𝐹(𝐳;𝑊,𝑏) = 𝐖𝐿⋅𝜎(𝐖𝐿−1 ⋯𝜎(𝐖1𝐳+ 𝐛1) ⋯+ 𝐛𝐿−1) + 𝐛𝐿,
(32)
in which (𝐖, 𝐛) represents all the parameters in the neural network, with 𝐖= (𝐖1, 𝐖2, … , 𝐖𝐿)
and 𝐛= (𝐛1, 𝐛2, … , 𝐛𝐿). Here 𝐖𝑙∈ℝ𝑛𝑙×𝑛𝑙−1 and 𝐛𝑙∈ℝ𝑛𝑙×1 for 𝑙= 1, 2, … , 𝐿, where 𝑛0 = 𝑛is
set as the dimension of the input variable. The operator 𝜎(⋅) takes a vector of any dimension
as input, and applies a function component-wise. Specifically, for any 𝑞∈ℤ+ and any vector
𝐮= (𝑢1, 𝑢2, … , 𝑢𝑞)⊤∈ℝ𝑞, we have that
𝜎(𝐮) = (𝜎(𝑢1), 𝜎(𝑢2), … , 𝜎(𝑢𝑞))⊤.

In the neural networks literature, the 𝐖𝑙’s are often called the weight matrices, the 𝐛𝑙’s are called
bias vectors, and 𝜎(⋅) is referred to as the activation function. Several popular choices for the acti-
vation function include ReLU with 𝜎(𝑢) = max(𝑢, 0), Leaky ReLU with 𝜎(𝑢) = 𝑎1 max(𝑢, 0) −
𝑎2 max(−𝑢, 0) where 𝑎1, 𝑎2 > 0, and smooth functions such as 𝜎(⋅) = tanh(⋅). In Equation (32),
information propagates from the input layer to the output layer in a feed-forward manner in the
sense that connections between the nodes do not form a cycle. Hence Equation (32) is also referred
to as fully-connected feed-forward neural network in the deep learning literature.
CNNs.
In addition to the FNNs above, CNNs are another type of feed-foward neural network that are
especially popular for image processing. In the finance setting, CNNs have been successfully
applied to price prediction based on inputs, which are images containing visualizations of price
dynamics and trading volumes (Jiang et al., 2020). The CNNs have two main building blocks—
convolutional layers and pooling layers. Convolutional layers are used to capture local patterns in
the images and pooling layers are used to reduce the dimension of the problem and improve the
computational efficiency.
Each convolutional layer uses a number of trainable filters to extract features from the data. We
start with the simple case of a single filter 𝐻, which applies the following convolution operation
𝑧∗𝐻∶ℝ𝑛𝑥×𝑛𝑦× ℝ𝑘𝑥×𝑘𝑦→ℝ(𝑛𝑥−𝑘𝑥+1)×(𝑛𝑦−𝑘𝑦+1) to the input 𝑧through
[𝑧∗𝐻]𝑖,𝑗=
𝑘𝑥
∑
𝑖′=1
𝑘𝑦
∑
𝑗′=1
[𝑧]𝑖+𝑖′−1,𝑗+𝑗′−1[𝐻]𝑖′,𝑗′.
The output of the convolutional layer ˆ𝑧is followed by the activation function 𝜎(⋅), that is
[ˆ𝑧]𝑖,𝑗= 𝜎([𝑧∗𝐻]𝑖,𝑗).
The weights and bias introduced for FNNs can also be incorporated in this setting thus, in
summary, the above simple CNN can be represented by
𝐹(𝑧;𝐻,𝑊,𝑏) = 𝑊⋅𝜎(𝑧∗𝐻) + 𝑏.
This simple convolutional layer can be generalized to the case of multiple channels with multiple
filters, where the input is a 3D tensor 𝑧∈ℝ𝑛𝑥×𝑛𝑦×𝑛𝑧where 𝑛𝑧denotes the number of channels.
Each channel represents a feature of the input, for example, an image as an input typically has
three channels, red, green, and blue. Then each filter is also represented by a 3D tensor 𝐻𝑘∈
ℝ𝑘𝑥×𝑘𝑦×𝑛𝑧(𝑘= 1, … , 𝐾) with the same third dimension 𝑛𝑧as the input and 𝐾is the total number
of filters. In this case, the convolution operation becomes
[𝑧∗𝐻𝑘]𝑖,𝑗=
𝑘𝑥
∑
𝑖′=1
𝑘𝑦
∑
𝑗′=1
𝑛𝑧
∑
𝑙=1
[𝑧]𝑖+𝑖′−1,𝑗+𝑗′−1,𝑙[𝐻]𝑖′,𝑗′,𝑙,
and the output is given by
[ˆ𝑧]𝑖,𝑗,𝑘= 𝜎([𝑧∗𝐻𝑘]𝑖,𝑗).

Pooling layers are used to aggregate the information and reduce the computational cost, typ-
ically after the convolutional layers. A commonly used pooling layer is the max pooling layer,
which computes the maximum value of a small neighborhood in the spatial coordinates. Note
that in addition, fully connected layers, as introduced above, are often used in the last few layers
of a CNN.
Recurrent Neural Networks (RNNs).
RNNs are a family of neural networks that are widely used in processing sequential data, including
speech, text, and financial time series data. Unlike feed-forward neural networks, RNNs are a class
of artificial neural networks where connections between units form a directed cycle. RNNs can use
their internal memory to process arbitrary sequences of inputs and hence are applicable to tasks
such as sequential data processing.
For RNNs, our input is a sequence of data 𝑧1,𝑧2, … ,𝑧𝑇. An RNN models the internal state ℎ𝑡
by a recursive relation
ℎ𝑡= 𝐹(ℎ𝑡−1,𝑧𝑡; 𝜃),
where 𝐹is a neural network with parameter 𝜃(for example 𝜃= (𝑊,𝑏)). Then the output is given
by
ˆ𝑧𝑡= 𝐺(ℎ𝑡−1,𝑧𝑡; 𝜙),
where 𝐺is another neural network with parameter 𝜙(𝜙can be the same as 𝜃). There are two
important variants of the vanilla RNN introduced above—the long short-term memory (LSTM)
and the gated recurrent units (GRUs). Compared to vanilla RNNs, LSTM and GRUs are shown
to have better performance in handling sequential data with long-term dependence due to their
flexibility in propagating information flows. Here we introduce the LSTM. Let ⊙denote element-
wise multiplication. The LSTM network architecture is given by
𝑓𝑡= 𝜎(𝑈𝑓𝑧𝑡+ 𝑊𝑓ℎ𝑡−1 + 𝑏𝑓)
𝑔𝑡= 𝜎(𝑈𝑔𝑧𝑡+ 𝑊𝑔ℎ𝑡−1 + 𝑏𝑔)
𝑞𝑡= 𝜎(𝑈𝑞𝑧𝑡+ 𝑊𝑞ℎ𝑡−1 + 𝑏𝑞)
𝑐𝑡= 𝑓𝑡⊙𝑐𝑡−1 + 𝑔𝑡⊙𝜎(𝑈𝑧𝑡+ 𝑊ℎ𝑡−1 + 𝑏)
ℎ𝑡= tanh(𝑐𝑡) ⊙𝑞𝑡
where 𝑈,𝑈𝑓,𝑈𝑔,𝑈𝑞,𝑊,𝑊𝑓,𝑊𝑔,𝑊𝑞are trainable weights matrices, and 𝑏,𝑏𝑓,𝑏𝑔,𝑏𝑞are train-
able bias vectors. In addition to the internal state ℎ𝑡, the LSTM also uses the gates and a cell state
𝑐𝑡. The forget gate 𝑓𝑡and the input gate 𝑔𝑡determine how much information can be transmitted
from the previous cell state𝑐𝑡−1 and from the update 𝜎(𝑈𝑧𝑡+ 𝑊ℎ𝑡−1 + 𝑏), to the current cell state
𝑐𝑡. The output gate 𝑞𝑡controls how much 𝑐𝑡reveals to ℎ𝑡. For more details, see Sutskever et al.
(2011) and Fan et al. (2021).

Training of neural networks.
Mini-batch stochastic gradient descent is a popular choice for training neural networks due to its
sample and computational efficiency. In this approach, the parameters 𝜃= (𝑊,𝑏) are updated
in the descent direction of an objective function (𝜃) by selecting a mini-batch of samples at
random to estimate the gradient of the objective function ∇𝜃(𝜃) with respect to the parameter 𝜃.
For example, in supervised learning, we aim to learn the relationship between an input 𝑋and an
output 𝑌, and the objective function (𝜃) measures the distance between the model prediction
(output) and the actual observation 𝑌. Assume that the dataset contains 𝑀samples of (𝑋, 𝑌).
Then the mini-batch stochastic gradient descent method is given as
𝜃(𝑛+1) = 𝜃(𝑛) −𝛽
ˆ
∇𝜃(𝜃(𝑛)),
𝑛= 0, … , 𝑁−1,
(33)
where 𝛽is a constant learning rate and
ˆ
∇𝜃(𝜃(𝑛)) is an estimate of the true gradient ∇𝜃(𝜃(𝑛)),
which is computed by averaging over 𝑚(𝑚≪𝑀) samples of (𝑋, 𝑌). It is called (vanilla) stochastic
gradient descent when 𝑚= 1, and it is the (traditional) gradient descent when 𝑚= 𝑀. Com-
pared with gradient descent, mini-batch stochastic gradient descent is noisy but computationally
efficient since it only uses a subset of the data, which is advantageous when dealing with large
datasets. It is also worth noting that, in addition to the standard mini-batch stochastic gradient
descent methods (33), momentum methods are popular extensions which take into account the
past gradient updates in order to accelerate the learning process. We give the updating rules for
two examples of gradient descent methods with momentum—the standard momentum and the
Nesterov momentum Nesterov (1983)
⎧
⎪
⎨
⎪⎩
𝑧𝑛+1 = 𝛽𝑧𝑛+ ∇𝜃(𝜃(𝑛))
⏟⎴⎴⎴⎴⎴⎴⎴⏟⎴⎴⎴⎴⎴⎴⎴⏟
the standard momentum
or
𝑧𝑛+1 = 𝛽𝑧𝑛+ ∇𝜃(𝜃(𝑛) −𝛼𝑧𝑛)
⏟⎴⎴⎴⎴⎴⎴⎴⎴⎴⏟⎴⎴⎴⎴⎴⎴⎴⎴⎴⏟
the Nesterov momentum
𝜃(𝑛+1) = 𝜃(𝑛) −𝛼𝑧𝑛+1,
where 𝛼and 𝛽are constant learning rates. Such methods are particularly useful when the
algorithms enter into a region where the gradient changes dramatically and thus the learned
parameters can bounce around the region which slows down the progress of the search. Addi-
tionally, there are many other variants such as RMSprop (Tieleman ... Hinton, 2012) and ADAM
(Kingma ... Ba, 2015), both of which employ adaptive learning rates.
## 3.2 Deep value-based RL algorithms
In this section, we introduce several 𝑄-learning algorithms with neural network approximations.
We refer the reader to François-Lavet et al. (2018) for other deep value-based RL algorithms such
as Neural TD learning and Dueling 𝑄-Networks.
Neural Fitted 𝑄-learning.
Fitted 𝑄-learning (Gordon, 1996) is a generalization of the classical 𝑄-learning algorithm with
functional approximations and it is applied in an off-line setting with a precollected dataset in
the form of tuples (𝑠, 𝑎, 𝑟, 𝑠′) with 𝑠′ ∼𝑃(𝑠, 𝑎) and 𝑟= 𝑟(𝑠, 𝑎), which is random. When the class
of approximation functionals is constrained to neural networks, the algorithm is referred to as
Neural Fitted 𝑄-learning and the 𝑄-function is parameterized by 𝑄(𝑠, 𝑎; 𝜃) = 𝐹((𝑠, 𝑎); 𝜃) with 𝐹

a neural network function parameterized by 𝜃(Riedmiller, 2005). For example, 𝐹can be set as
Equation (32) with 𝜃= (𝑊,𝑏).
In Neural Fitted 𝑄-learning, the algorithm starts with some random initialization of the 𝑄-
values 𝑄(𝑠, 𝑎; 𝜃(0)) where 𝜃(0) refers to the initial parameters. Then, an approximation of the 𝑄-
values at the 𝑛th iteration 𝑄(𝑠, 𝑎; 𝜃(𝑛)) is updated towards the target value
𝑌𝑄
𝑛= 𝑟+ 𝛾max
𝑎′∈𝑄(𝑠′, 𝑎′; 𝜃(𝑛)),
(34)
where 𝜃(𝑛) are the neural network parameters in the 𝑛th iteration, updated by stochastic gradient
descent (or a variant) by minimizing the square loss:
𝑁𝐹𝑄(𝜃(𝑛)) = ‖𝑄(𝑠, 𝑎; 𝜃(𝑛)) −𝑌𝑄
𝑛‖2
2.
(35)
Thus, the 𝑄-learning update amounts to updating the parameters:
𝜃(𝑛+1) = 𝜃(𝑛) + 𝛽
(
𝑌𝑄
𝑛−𝑄(𝑠, 𝑎; 𝜃(𝑛))
)
∇𝜃(𝑛)𝑄(𝑠, 𝑎; 𝜃(𝑛))
(36)
where 𝛽is a learning rate. This update resembles stochastic gradient descent, updating the current
value 𝑄(𝑠, 𝑎; 𝜃(𝑛)) towards the target value 𝑌𝑄
𝑛.
When neural networks are applied to approximate the 𝑄-function, it has been empirically
observed that Neural Fitted 𝑄-learning may suffer from slow convergence or even divergence
Riedmiller (2005). In addition, the approximated 𝑄-values tend to be overestimated due to the
max operator (Van Hasselt et al., 2016).
Deep 𝑄-Network (DQN).
To overcome the instability issue and the risk of overestimation mentioned above, Mnih et al.
(2015) proposed a DQN algorithm in an online setting with two novel ideas. One idea is the slow-
update of the target network and the second is the use of “experience replay.” Both these ideas
dramatically improve the empirical performance of the algorithm and DQN has been shown to
have a strong performance for a variety of ATARI games (Bellemare et al., 2013).
We first discuss the use of experience replay (Lin, 1992). That is we introduce a replay memory
. At each time 𝑡, the tuple (𝑠𝑡, 𝑎𝑡, 𝑟𝑡, 𝑠𝑡+1) is stored in and a mini-batch of 𝐵independent sam-
ples is randomly selected from to train the neural network via stochastic gradient descent. Since
the trajectory of an MDP has strong temporal correlation, the goal of experience replay is to obtain
uncorrelated samples that are more similar to the i.i.d data (often assumed in many optimization
algorithms), which can give more accurate gradient estimation for the stochastic optimization
problem and enjoy better convergence performance. For experience replay, the replay memory
size is usually very large in practice. For example, the replay memory size is 106 in Mnih et al.
(2015). Moreover, DQN uses the 𝜀-greedy policy, which enables exploration over the state-action
space × . Thus, when the replay memory is large, experience replay is close to sampling inde-
pendent transitions from an explorative policy. This reduces the variance of the gradient, which
is used to update 𝜃. Thus, experience replay stabilizes the training of DQN, which benefits the
algorithm in terms of computational efficiency.
We now discuss using a target network 𝑄(⋅, ⋅; 𝜃−) with parameter 𝜃−(the current estimate of
the parameter). With independent samples {(𝑠(𝑖), 𝑎(𝑖), 𝑟(𝑖), 𝑠′
(𝑖))}𝐵
𝑖=0 from the replay memory (we

use 𝑠′
(𝑖) instead of 𝑠(𝑖+1) for the state after (𝑠(𝑖), 𝑎(𝑖)) to avoid notational confusion with the next
independent sample 𝑠(𝑖+1) in the state space), to update the parameter 𝜃of the 𝑄-network, we
compute the target
˜𝑌𝑄
𝑖= 𝑟(𝑖) + 𝛾max
𝑎′∈𝑄(𝑠′
(𝑖), 𝑎′; 𝜃(𝑛)−),
(37)
and update 𝜃using the gradient of
𝐷𝑄𝑁(𝜃(𝑛), 𝜃(𝑛)−) = 1
𝐵
𝐵
∑
𝑖=1
‖‖‖𝑄(𝑠(𝑖), 𝑎(𝑖); 𝜃(𝑛)) −˜𝑌𝑄
𝑖
‖‖‖
2
2.
(38)
Whereas the parameter 𝜃(𝑛)−is updated once every 𝑇target steps by letting 𝜃(𝑛)−= 𝜃(𝑛), if 𝑛=
𝑚𝑇target for some 𝑚∈ℤ+, and 𝜃(𝑛)−= 𝜃(𝑛−1)−otherwise. That is, the target network is held fixed
for 𝑇target steps and then updated using the current weights of the 𝑄-network. The introduction
of a target network prevents the rapid propagation of instabilities and it reduces the risk of diver-
gence. The idea of target networks can be seen as an instantiation of Fitted 𝑄-learning, where
each period between target network updates corresponds to a single Fitted 𝑄-iteration.
Double Deep 𝑄-network (DQN).
The max operator in Neural Fitted 𝑄-learning and DQN, in Equations (34) and (37), uses the
same values both to select and to evaluate an action. This makes it more likely to select overes-
timated values, resulting in over optimistic value estimates. To prevent this, Double 𝑄-learning
(Hasselt, 2010) decouples the selection from the evaluation and this is further extended to the
neural network setting (Van Hasselt et al., 2016).
In double 𝑄-learning (Hasselt, 2010) and double deep 𝑄-network (Van Hasselt et al., 2016), two
value functions are learned by assigning experiences randomly to update one of the two value
functions, resulting in two sets of weights, 𝜃and 𝜂. For each update, one set of weights is used to
determine the greedy policy and the other to determine its value. For a clear comparison, we can
untangle the selection and evaluation in Neural Fitted 𝑄-learning and rewrite its target (34) as
𝑌𝑄
𝑛= 𝑟+ 𝛾𝑄(𝑠′, arg max
𝑎∈𝑄(𝑠′, 𝑎; 𝜃(𝑛)); 𝜃(𝑛)).
(39)
The target of Double (deep) 𝑄-learning can then be written as
𝑌
𝑄
𝑛= 𝑟+ 𝛾𝑄(𝑠′, arg max
𝑎∈𝑄(𝑠′, 𝑎; 𝜃(𝑛)); 𝜂(𝑛)).
(40)
Notice that the selection of the action, in the argmax, is still due to the online weights 𝜃(𝑛). This
means that, as in 𝑄-learning, we are still estimating the value of the greedy policy according to the
current values, as defined by 𝜃(𝑛). However, we use the second set of weights 𝜂(𝑛) to fairly compute
the value of this policy. This second set of weights can be updated symmetrically by switching the
roles of 𝜃and 𝜂.
Convergence guarantee.
For DQN, Fan et al. (2020) characterized the approximation error of the 𝑄-function by the sum
of a statistical error and an algorithmic error, and the latter decays to zero at a geometric rate as

the algorithm proceeds. The statistical error characterizes the bias and variance arising from the
𝑄-function approximation using the neural network. Cai et al. (2019) parametrized the 𝑄-function
by a two-layer neural network and provided a mean-squared sample complexity with sublinear
convergence rate for neural TD learning. The two-layer network in Cai et al. (2019) with width 𝑚
is given by
𝐹((𝑠, 𝑎);𝑊) =
1
√
𝑚
𝑚
∑
𝑙=1
𝑐𝑙⋅𝜎
(
𝑊⊤
𝑙(𝑠, 𝑎)
)
,
(41)
where the activation function 𝜎(⋅) is set to be the ReLU function, and the parameter𝑐= (𝑐1, … , 𝑐𝑚)
is fixed at the initial parameter during the training process and only the weights 𝑊are updated.
Xu and Gu (2020) considered a more challenging setting than Cai et al. (2019), where the input
data are non-i.i.d and the neural network has multiple (more than two) layers and obtained the
same sublinear convergence rate. Furthermore, Cayci et al. (2021) also employed the two-layer
neural network in Equation (41) and proved that two algorithms used in practice, the projection-
free and max-norm regularized (Goodfellow et al., 2013) neural TD, achieve mean-squared sample
complexity of ˜′(1∕𝜀6) and ˜′(1∕𝜀4), respectively.
## 3.3 Deep policy-based RL algorithms
In this section, we focus on deep policy-based methods, which are extensions of policy-based
methods using neural network approximations. We parameterize the policy 𝜋by a neural net-
work 𝐹with parameter 𝜃= (𝑊,𝑏), that is, 𝑎∼𝜋(𝑠, 𝑎; 𝜃) = 𝑓(𝐹((𝑠, 𝑎); 𝜃)) for some function 𝑓. A
popular choice of 𝑓is given by
𝑓(𝐹((𝑠, 𝑎); 𝜃)) =
exp(𝜏𝐹((𝑠, 𝑎); 𝜃))
∑
𝑎′∈exp(𝜏𝐹((𝑠, 𝑎′); 𝜃)),
for some parameter 𝜏, which gives an energy-based policy (see, e.g., Haarnoja et al., 2017; Wang
et al., 2020). The policy parameter 𝜃is updated using the gradient ascent rule given by
𝜃(𝑛+1) = 𝜃(𝑛) + 𝛽
ˆ
∇𝜃𝐽(𝜃(𝑛)),
𝑛= 0, … , 𝑁−1,
where
ˆ
∇𝜃𝐽(𝜃(𝑛)) is an estimate of the policy gradient.
Using neural networks to parametrize the policy and/or value functions in the vanilla version
of policy-based methods discussed in Section 2.4 leads to neural Actor–Critic algorithms (Wang
et al., 2020), neural PPO/TRPO (Liu et al., 2019), and deep DPG (DDPG) (Lillicrap et al., 2016).
In addition, since introducing an entropy term in the objective function encourages policy
exploration (Haarnoja et al., 2017) and speeds the learning process (Haarnoja et al., 2018; Mei
et al., 2020) (as discussed in Section 2.5.4), there have been some recent developments in
(off-policy) soft Actor–Critic algorithms (Haarnoja et al., 2018), (Haarnoja et al., 2018) using
neural networks, which solve the RL problem with entropy regularization. Below we introduce
the DDPG algorithm, which is one of the most popular deep policy-based methods, and which
has been applied in many financial problems.

A L G O R I T H M 6 The DDPG algorithm
DDPG.
DDPG is a model-free off-policy Actor–Critic algorithm, first introduced in Lillicrap et al. (2016),
which combines the DQN and DPG algorithms. Since its structure is more complex than DQN and
DPG, we provide the pseudocode for DDPG in Algorithm 6. DDPG extends the DQN to continuous
action spaces by incorporating DPG to learn a deterministic strategy. To encourage exploration,
DDPG uses the following action:
𝑎𝑡∼𝜋𝐷(𝑠𝑡; 𝜃𝑡) + 𝜖,
where 𝜋𝐷is a deterministic policy and 𝜖is a random variable sampled from some distribution
, which can be chosen according to the environment. Note that the algorithm requires a small
learning rate ̄𝛽≪1 (see line 14 in Algorithm 6) to improve the stability of learning the target
networks. In the same way as DQN (see Section 3.2), the DDPG algorithm also uses a replay buffer
to improve the performance of neural networks.
Convergence guarantee.
By parameterizing the policy and/or value functions using a two-layer neural network given in
Equation (41), Liu et al. (2019) provided a mean-squared sample complexity for neural PPO and
TRPO algorithms with sublinear convergence rate; Wang et al. (2020) studied neural Actor–Critic
methods where the actor updates using (1) vanilla policy gradient or (2) natural policy gradient,
and in both cases, the critic updates using TD(0). They proved that in case (1), the algorithm
converges to a stationary point at a sublinear rate and they also established the global optimality
of all stationary points under mild regularity conditions. In case (2), the algorithm was proved to

achieve a mean-squared sample complexity with sublinear convergence rate. To the best of our
knowledge, no convergence guarantee has been established for the DDPG (and DPG) algorithms.
## 4 Applications In Finance
The availability of data from electronic markets and the recent developments in RL together have
led to a rapidly growing recent body of work applying RL algorithms to decision-making problems
in electronic markets. Examples include optimal execution, portfolio optimization, option pricing
and hedging, market making, order routing, as well as robot advising.
In this section, we start with a brief overview of electronic markets and some discussion of
market microstructure in Section 4.1. We then introduce several applications of RL in finance. In
particular, optimal execution for a single asset is introduced in Section 4.2 and portfolio optimiza-
tion problems across multiple assets is discussed in Section 4.3. This is followed by sections on
option pricing, robo-advising, and SOR. In each, we introduce the underlying problem and basic
model before looking at recent RL approaches used in tackling them.
It is worth noting that there are some open source projects that provide full pipelines for
implementing different RL algorithms in financial applications (Liu et al., 2021, 2022).
4.1
Preliminary: Electronic markets and market microstructure
Many recent decision-making problems in finance are centered around electronic markets. We
give a brief overview of this type of market and discuss two popular examples—central limit order
books (LOBs) and electronic over-the-counter (OTC) markets. For more in-depth discussion of
electronic markets and market microstructure, we refer the reader to the books (Cartea et al.,
2015) and (Lehalle ... Laruelle, 2018).
Electronic markets.
Electronic markets have emerged as popular venues for the trading of a wide variety of financial
assets. Stock exchanges in many countries, including Canada, Germany, Israel, and the United
Kingdom, have adopted electronic platforms to trade equities, as has Euronext, the market com-
bining several former European stock exchanges. In the United States, electronic communications
networks (ECNs) such as Island, Instinet, and Archipelago (now ArcaEx) use an electronic order
book structure to trade as much as 45% of the volume on the NASDAQ stock market. Several
electronic systems trade corporate bonds and government bonds (e.g., BrokerTec, MTS), while
in foreign exchange, electronic systems such as EBS and Reuters dominate the trading of cur-
rencies. Eurex, the electronic Swiss-German exchange, is now the world’s largest futures market,
while options have been traded in electronic markets since the opening of the International Secu-
rities Exchange in 2000. Many such electronic markets are organized as electronic LOBs. In this
structure, there is no designated liquidity provider such as a specialist or a dealer. Instead, liq-
uidity arises endogenously from the submitted orders of traders. Traders who submit orders to
buy or sell the asset at a particular price are said to “make” liquidity, while traders who choose to
hit existing orders are said to “take” liquidity. Another major type of electronic market without a
central LOB is the OTC market where the orders are executed directly between two parties, the
dealer and the client, without the supervision of an exchange. Examples include BrokerTec and
MTS for trading corporate bonds and government bonds.

F I G U R E 1
A snapshot of the LOB of MSFT(Microsoft) stock at 9:30:08.026 a.m. on June 21, 2012 with 10
levels of bid/ask prices. [Color figure can be viewed at wileyonlinelibrary.com]
LOBs.
An LOB is a list of orders that a trading venue, for example, the NASDAQ exchange, uses to record
the interest of buyers and sellers in a particular financial asset or instrument. There are two types
of orders the buyers (sellers) can submit: a limit buy (sell) order with a preferred price for a given
volume of the asset or a market buy (sell) order with a given volume, which will be immediately
executed with the best available limit sell (buy) orders. Therefore, limit orders have a price guar-
antee but are not guaranteed to be executed, whereas market orders are executed immediately
at the best available price. The lowest price of all sell limit orders is called the ask price, and the
highest price of all buy limit orders is called the bid price. The difference between the ask and bid
is known as the spread and the average of the ask and bid is known as the mid-price. A snapshot
of an LOB4 with 10 levels of bid/ask prices is shown in Figure 1.
A matching engine is used to match the incoming buy and sell orders. This typically follows
the price-time priority rule (Preis, 2011), whereby orders are first ranked according to their price.
Multiple orders having the same price are then ranked according to the time they were entered. If
the price and time are the same for the incoming orders, then the larger order gets executed first.
The matching engine uses the LOB to store pending orders that could not be executed upon arrival.
OTC markets.
OTC or off-exchange trading is done directly between two parties, without the supervision of an
exchange. Many OTC markets organized around dealers, including corporate bond markets in
many countries, have also undergone upheaval linked to electronification over the last 10 years.
The electronification process is dominated by Multi-dealer-to-client (MD2C) platforms enabling
clients to send the same request for a quote (RFQ) to several dealers simultaneously and therefore
instantly put the dealers into competition with one another. These dealers can then each provide
a price to the client for the transaction (not necessarily the price the dealer has streamed). Dealers
4 We use NASDAQ ITCH data taken from Lobster https://lobsterdata.com/.

know the identity of the client (which differs from most of the systems organized around a central
LOB) and the number of requested dealer prices. However, they do not see the prices that are
streamed by the other dealers. They only see a composite price at the bid and offer, based on some
of the best streamed prices. The client progressively receives the answers to the RFQ and can deal
at any time with the dealer who has proposed the best price, or decide not to trade. Each dealer
knows whether a deal was done (with her/him, but also with another dealer—without knowing
the identity of this dealer) or not. If a transaction occurred, the best dealer usually knows the cover
price (the second best bid price in the RFQ), if there is one. We refer the reader to Fermanian et al.
(2015) for a more in-depth discussion of MD2C bond trading platforms.
Market participants.
When considering different market participants, it is sometimes helpful to classify them based on
their objectives and trading strategies. Three primary classes are the following Cartea et al. (2015):
∙Fundamental (or noise or liquidity) traders: those who are driven by economic fundamentals
outside the exchange.
∙Informed traders: traders who profit from leveraging information not reflected in market prices
by trading assets in anticipation of their appreciation or depreciation.
∙Market makers: professional traders who profit from facilitating exchange in a particular asset
and exploit their skills in executing trades.
The effects of the interactions amongst all these traders is one of the key issues studied in the
field of market microstructure. How to improve trading decisions from the perspective of one
class of market participants, while strategically interacting with the others, is one of the big chal-
lenges in the field. The recent literature has seen many attempts to exploit RL techniques to tackle
these problems.
## 4.2 Optimal execution
Optimal execution is a fundamental problem in financial modeling. The simplest version is the
case of a trader who wishes to buy or sell a given amount of a single asset within a given time
period. The trader seeks strategies that maximize their return from, or alternatively, minimize the
cost of, the execution of the transaction.
The Almgren–Chriss model.
A classical framework for optimal execution is due to Almgren and Chriss (2001). In this setup, a
trader is required to sell an amount 𝑞0 of an asset, with price 𝑆0 at time 0, over the time period [0, 𝑇]
with trading decisions made at discrete time points 𝑡= 1, … , 𝑇. The final inventory 𝑞𝑇is required
to be zero. Therefore, the goal is to determine the liquidation strategy 𝑢1, 𝑢2, … , 𝑢𝑇, where 𝑢𝑡(𝑡=
1, 2, … , 𝑇) denotes the amount of the asset to sell at time 𝑡. It is assumed that selling quantities of
the asset will have two types of price impact—a temporary impact which refers to any temporary
price movement due to the supply-demand imbalance caused by the selling, and a permanent
impact, which is a long-term effect on the “equilibrium” price due to the trading, that will remain
at least for the trading period.

We write 𝑆𝑡for the asset price at time 𝑡. The Almgren–Chriss model assumes that the asset
price evolves according to the discrete arithmetic random walk
𝑆𝑡= 𝑆𝑡−1 + 𝜎𝜉𝑡−𝑔(𝑢𝑡),
𝑡= 1, 2, … , 𝑇
where 𝜎is the (constant) volatility parameter, 𝜉𝑡are independent random variables drawn from a
distribution with zero mean and unit variance, and 𝑔is a function of the trading strategy 𝑢𝑡that
measures the permanent impact. The inventory process {𝑞𝑡}𝑇
𝑡=0 records the current holding in the
asset, 𝑞𝑡, at each time 𝑡. Thus we have
𝑞𝑡= 𝑞𝑡−1 −𝑢𝑡.
Selling the asset may cause a temporary price impact, that is a drop in the average price per share,
thus the actual price per share received is
˜𝑆𝑡= 𝑆𝑡−1 −ℎ(𝑢𝑡),
where ℎis a function which quantifies this temporary price impact. The cost of this trading tra-
jectory is defined to be the difference between the initial book value and the revenue, given by
𝐶= 𝑞0𝑆0 −∑𝑇
𝑡=1 𝑞𝑡˜𝑆𝑡with mean and variance
𝔼[𝐶] =
𝑇
∑
𝑡=1
(𝑞𝑡𝑔(𝑢𝑡) + 𝑢𝑡ℎ(𝑢𝑡)),
Var(𝐶) = 𝜎2
𝑇
∑
𝑡=1
𝑞2
𝑡.
A trader, thus, would like to minimize her expected cost as well as the variance of the cost, which
gives the optimal execution problem in this model as
min
{𝑢𝑡}𝑇
𝑡=1
(𝔼[𝐶] + 𝜆Var(𝐶)),
where 𝜆∈ℝis a measure of risk-aversion.
When we assume that both the price impacts are linear, that is
𝑔(𝑢𝑡) = 𝛾𝑢𝑡,
ℎ(𝑢𝑡) = 𝜂𝑢𝑡,
where 𝛾and 𝜂are the permanent and temporary price impact parameters, Almgren and Chriss
(2001) derive the general solution for the Almgren–Chriss model. This is given by
𝑢𝑗=
2 sinh
( 1
2𝜅
)
sinh(𝜅𝑇) cosh
(
𝜅
(
𝑇−𝑡𝑗−1
2
))
𝑞0,
𝑗= 1, 2, … , 𝑇,
with
𝜅= cosh−1
( ̃𝜅2
2 + 1
)
,
̃𝜅2 =
𝜆𝜎2
𝜂(1 −𝛾
2𝜂)
.

The corresponding optimal inventory trajectory is
𝑞𝑗=
sinh
(
𝜅(𝑇−𝑡𝑗)
)
sinh(𝜅𝑇)
𝑞0,
𝑗= 0, 1, … , 𝑇.
(42)
The above Almgren–Chriss framework for liquidating a single asset can be extended to the case of
multiple assets (Almgren ... Chriss, 2001, Appendix A). We also note that the general solution to
the Almgren-Chriss model had been constructed previously in Grinold and Kahn (2000, Chapter
16).
This simple version of the Almgren–Chriss framework has a closed-form solution but it relies
heavily on the assumptions of the dynamics and the linear form of the permanent and temporary
price impact. The mis-specification of the dynamics and market impacts may lead to undesirable
strategies and potential losses. In addition, the solution in Equation (42) is a preplanned strategy
that does not depend on real-time market conditions. Hence this strategy may miss certain oppor-
tunities when the market moves. This motivates the use of an RL approach, which is more flexible
and able to incorporate market conditions when making decisions.
Evaluation criteria and benchmark algorithms.
Before discussing the RL approach, we introduce several widely used criteria to evaluate the
performance of execution algorithms in the literature such as the Profit and Loss (PnL), Imple-
mentation Shortfall, and the Sharp ratio. The PnL is the final profit or loss induced by a given
execution algorithm over the whole time period, which is made up of transactions at all time
points. The Implementation Shortfall (Perold, 1988) for an execution algorithm is defined as the
difference between the PnL of the algorithm and the PnL received by trading the entire amount
of the asset instantly. The Sharpe ratio (Sharpe, 1966) is defined as the ratio of expected return
to standard deviation of the return; thus, it measures return per unit of risk. Two popular vari-
ants of the Sharpe ratio are the differential Sharpe ratio (Moody et al., 1998) and the Sortino ratio
(Sortino & Price, 1994). In addition, some classical prespecified strategies are used as benchmarks
to evaluate the performance of a given RL-based execution strategy. Popular choices include exe-
cutions strategies based on time-weighted average price (TWAP) and volume-weighted average
price (VWAP) as well as the Submit and Leave (SnL) policy where a trader places a sell order for
all shares at a fixed limit order price, and goes to the market with any unexecuted shares remaining
at time 𝑇.
RL approach.
We first provide a brief overview of the existing literature on RL for optimal execution. The most
popular types of RL methods that have been used in optimal execution problems are 𝑄-learning
algorithms and (double) DQN (Dabérius et al., 2019; Hendricks ... Wilcox, 2014; Jeong & Kim,
2019; Ning et al., 2018; Cartea et al., 2021; Shen et al., 2014; Nevmyvaka et al., 2006; Zhang et al.,
2020). Policy-based algorithms are also popular in this field, including (deep) policy gradient
methods (Hambly et al., 2021; Zhang et al., 2020), A2C (Zhang et al., 2020), PPO (Dabérius et al.,
2019; Lin & Beling, 2020), and DDPG (Ye et al., 2020). The benchmark strategies studied in these
papers include the Almgren–Chriss solution (Hendricks & Wilcox, 2014; Hambly et al., 2021), the
TWAP strategy (Ning et al., 2018; Dabérius et al., 2019; Lin & Beling, 2020), the VWAP strategy (Lin
& Beling, 2020), and the SnL policy (Nevmyvaka et al., 2006; Ye et al., 2020). In some models, the
trader is allowed to buy or sell the asset at each time point (Jeong & Kim, 2019; Zhang et al., 2020;
Wei et al., 2019; Deng et al., 2017), whereas there are also many models where only one trading

direction is allowed (Dabérius et al., 2019; Hendricks & Wilcox, 2014; Hambly et al., 2021; Nevmy-
vaka et al., 2006; Lin & Beling, 2020; Ning et al., 2018; Shen et al., 2014; Ye et al., 2020). The state
variables are often composed of time stamp, the market attributes including (mid-)price of the
asset and/or the spread, the inventory process and past returns. The control variables are typically
set to be the amount of asset (using market orders) to trade and/or the relative price level (using
limit orders) at each time point. Examples of reward functions include cash inflow or outflow
(depending on whether we sell or buy) (Nevmyvaka et al., 2006; Shen et al., 2014), implementa-
tion shortfall (Hendricks & Wilcox, 2014), profit (Deng et al., 2017), Sharpe ratio (Deng et al., 2017),
return (Jeong ... Kim, 2019), and PnL (Wei et al., 2019). Popular choices of performance measure
include Implementation Shortfall (Hendricks ... Wilcox, 2014; Hambly et al., 2021), PnL (with a
penalty term of transaction cost) (Deng et al., 2017; Ning et al., 2018; Wei et al., 2019), trading cost
(Nevmyvaka et al., 2006; Shen et al., 2014), profit (Deng et al., 2017; Jeong & Kim, 2019), Sharpe
ratio (Deng et al., 2017; Zhang et al., 2020), Sortino ratio (Zhang et al., 2020), and return (Zhang
et al., 2020).
We now discuss some more details of the RL algorithms and experimental settings in the
above papers. For value-based algorithms, Nevmyvaka et al. (2006) provided the first large scale
empirical analysis of a RL method applied to optimal execution problems. They focused on a mod-
ified 𝑄-learning algorithm to select price levels for limit order trading, which leads to significant
improvements over simpler forms of optimization such as the SnL policies in terms of trading
costs. Shen et al. (2014) proposed a risk-averse RL algorithm for optimal liquidation, which can
be viewed as a generalization of Nevmyvaka et al. (2006). This algorithm achieves substantially
lower trading costs over the period when the 2010 flash-crash happened compared with the risk-
neutral RL algorithm in Nevmyvaka et al. (2006). Hendricks and Wilcox (2014) combined the
Almgren–Chriss solution and the 𝑄-learning algorithm and showed that they are able to improve
the Implementation Shortfall of the Almgren–Chriss solution by up to 10.3% on average, using
LOB data, which includes five price levels. Ning et al. (2018) proposed a modified double DQN
algorithm for optimal execution and showed that the algorithm outperforms the TWAP strategy
on seven out of nine stocks using PnL as the performance measure. They added a single one-
second time step Δ𝑇at the end of the horizon to guarantee that all shares are liquidated over
𝑇+ Δ𝑇. Jeong and Kim (2019) designed a trading system based on DQN, which determines both
the trading direction and the number of shares to trade. Their approach was shown to increase
the total profits by at least four times in four different index stocks compared with a benchmark
trading model, which trades a fixed number of shares each time. They also used transfer learning
to avoid overfitting, where some knowledge/information is reused when learning in a related or
similar situation.
For policy-based algorithms, Deng et al. (2017) combined deep learning with RL to determine
whether to sell, hold, or buy at each time point. In the first step of their model, neural networks
are used to summarize the market features and in the second step, the RL part makes trading deci-
sions. The proposed method was shown to outperform several other deep learning and deep RL
models in terms of PnL, total profits, and Sharpe ratio. They suggested that in practice, the Sharpe
ratio was more recommended as the reward function compared with total profits. Ye et al. (2020)
used the DDPG algorithm for optimal execution over a short time horizon (two minutes) and
designed a network to extract features from the market data. Experiments on real LOB data with
10 price levels show that the proposed approach significantly outperforms the existing methods,
including the SnL policy (as baseline), the 𝑄-learning algorithm, and the method in Hendricks
and Wilcox (2014). Lin and Beling (2020) proposed an adaptive framework based on PPO with
neural networks including LSTM and fully connected networks, and showed that the framework

outperforms the baseline models including TWAP and VWAP, as well as several deep RL models
on most of 14 US equities. Hambly et al. (2021) applied the (vanilla) policy gradient method to the
LOB data of five stocks in different sectors and showed that they improve the Implementation
Shortfall of the Almgren–Chriss solution by around 20%. Leal et al. (2020) used neural networks
to learn the mapping between the risk-aversion parameter and the optimal control with potential
market impacts incorporated.
For comparison between value- and policy-based algorithms, Dabérius et al. (2019) explored
double DQN and PPO algorithms in different market environments—when the benchmark TWAP
is optimal, PPO is shown to converge to TWAP whereas double DQN may not; when TWAP
is not optimal, both algorithms outperform this benchmark. Zhang et al. (2020) showed that
DQN, policy gradient, and A2C outperform several baseline models including classical time-
series momentum strategies, on test data of 50 liquid futures contracts. Both continuous and
discrete action spaces were considered in their work. They observed that DQN achieves the best
performance and the second best is the A2C approach.
In addition, model-based RL algorithms have also been used for optimal execution. Wei et al.
(2019) built a profitable electronic trading agent to place buy and sell orders using model-based
RL, which outperforms two benchmarks strategies in terms of PnL on LOB data. They used a
recurrent neural network to learn the state transition probability. We note that multi-agent RL has
also been used to address the optimal execution problem, see, for example, Bao and Liu (2019);
Karpe et al. (2020).
## 4.3 Portfolio optimization
In portfolio optimization problems, a trader needs to select and trade the best portfolio of assets
in order to maximize some objective function, which typically includes the expected return and
some measure of the risk. The benefit of investing in such portfolios is that the diversification of
investments achieves higher return per unit of risk than only investing in a single asset (see, e.g.,
Zivot, 2017).
Mean–variance portfolio optimization.
The first significant mathematical model for portfolio optimization is the Markowitz model
(Markowitz, 1952), also called the mean–variance model, where an investor seeks a portfolio to
maximize the expected total return for any given level of risk measured by variance. This is a
single-period optimization problem and is then generalized to multiperiod portfolio optimization
problems in Mossin (1968), Hakansson (1971), Samuelson (1975), Merton and Samuelson (1974),
Steinbach (2001), Li and Ng (2000). In this mean-variance framework, the risk of a portfolio is
quantified by the variance of the wealth and the optimal investment strategy is then sought to
maximize the final wealth penalized by a variance term. The mean–variance framework is of
particular interest because it not only captures both the portfolio return and the risk, but also
suffers from the time-inconsistency problem (Vigna, 2016; Strotz, 1955; Xiao et al., 2020). That is
the optimal strategy selected at time 𝑡is no longer optimal at time 𝑠> 𝑡and the Bellman equa-
tion does not hold. A breakthrough was made in Li and Ng (2000), in which they were the first
to derive the analytical solution to the discrete-time multiperiod mean–variance problem. They
applied an embedding approach, which transforms the mean–variance problem to an LQ problem
where classical approaches can be used to find the solution. The same approach was then used to
solve the continuous-time mean–variance problem (Zhou & Li, 2000). In addition to the embed-

ding scheme, other methods including the consistent planning approach (Basak ... Chabakauri,
2010; Bjork ... Murgoci, 2010) and the dynamically optimal strategy Pedersen and Peskir (2017)
have also been applied to solve the time-inconsistency problem arising in the mean–variance
formulation of portfolio optimization.
Here we introduce the multiperiod mean–variance portfolio optimization problem as given in
Li and Ng (2000). Suppose there are 𝑛risky assets in the market and an investor enters the market
at time 0 with initial wealth 𝑥0. The goal of the investor is to reallocate his wealth at each time point
𝑡= 0, 1, … , 𝑇among the 𝑛assets to achieve the optimal trade off between the return and the risk of
the investment. The random rates of return of the assets at 𝑡is denoted by 𝑒𝑡= (𝑒1
𝑡, … , 𝑒𝑛
𝑡)⊤, where
𝑒𝑖
𝑡(𝑖= 1, … , 𝑛) is the rate of return of the 𝑖th asset at time 𝑡. The vectors 𝑒𝑡, 𝑡= 0, 1, … , 𝑇−1, are
assumed to be statistically independent (this independence assumption can be relaxed, see, e.g.,
Xiao et al. (2020)) with known mean 𝜇𝑡= (𝜇1
𝑡, … , 𝜇𝑛
𝑡)⊤∈ℝ𝑛and known standard deviation 𝜎𝑖
𝑡
for 𝑖= 1, … , 𝑛and 𝑡= 0, … , 𝑇−1. The covariance matrix is denoted by Σ𝑡∈ℝ𝑛×𝑛, where [Σ𝑡]𝑖𝑖=
(𝜎𝑖
𝑡)2 and [Σ𝑡]𝑖𝑗= 𝜌𝑖𝑗
𝑡𝜎𝑖
𝑡𝜎𝑗
𝑡for 𝑖, 𝑗= 1, … , 𝑛and 𝑖≠𝑗, where 𝜌𝑖𝑗
𝑡is the correlation between assets
𝑖and 𝑗at time 𝑡. We write 𝑥𝑡for the wealth of the investor at time 𝑡and 𝑢𝑖
𝑡(𝑖= 1, … , 𝑛−1) for
the amount invested in the 𝑖th asset at time 𝑡. Thus the amount invested in the 𝑛th asset is 𝑥𝑡−
∑𝑛−1
𝑖=1 𝑢𝑖
𝑡. An investment strategy is denoted by 𝑢𝑡= (𝑢1
𝑡, 𝑢2
𝑡, … , 𝑢𝑛−1
𝑡
)⊤for 𝑡= 0, 1, … , 𝑇−1, and
the goal is to find an optimal strategy such that the portfolio return is maximized while minimizing
the risk of the investment, that is,
max
{𝑢𝑡}𝑇−1
𝑡=0
𝔼[𝑥𝑇] −𝜙Var(𝑥𝑇),
(43)
subject to
𝑥𝑡+1 =
𝑛−1
∑
𝑖=1
𝑒𝑖
𝑡𝑢𝑖
𝑡+
(
𝑥𝑡−
𝑛−1
∑
𝑖=1
𝑢𝑖
𝑡
)
𝑒𝑛
𝑡,
𝑡= 0, 1, … , 𝑇−1,
(44)
where 𝜙is a weighting parameter balancing risk, represented by the variance of 𝑥𝑇, and return.
As mentioned above, this framework is embedded into an LQ problem in Li and Ng (2000), and
the solution to the LQ problem gives the solution to the above problem (43)–(44). The analytical
solution derived in Li and Ng (2000) is of the following form:
𝑢∗
𝑡(𝑥𝑡) = 𝛼𝑡𝑥𝑡+ 𝛽𝑡,
where 𝛼𝑡and 𝛽𝑡are explicit functions of 𝜇𝑡and Σ𝑡, which are omitted here and can be found in
Li and Ng (2000).
The above framework has been extended in different ways, for example, the risk-free asset
can also be involved in the portfolio, and one can maximize the cumulative form of Equation
(43) rather than only focusing on the final wealth 𝑥𝑇. For more details about these variants
and solutions, see Xiao et al. (2020). In addition to the mean–variance framework, other major
paradigms in portfolio optimization are the Kelly Criterion and Risk Parity. We refer to Sato
(2019) for a review of these optimal control frameworks and popular model-free RL algorithms
for portfolio optimization.
Note that the classical stochastic control approach for portfolio optimization problems across
multiple assets requires both a realistic representation of the temporal dynamics of individual
assets, as well as an adequate representation of their co-movements. This is extremely difficult

when the assets belong to different classes (for example, stocks, options, futures, interest rates,
and their derivatives). On the other hand, the model-free RL approach does not rely on the
specification of the joint dynamics across assets.
RL approach.
Both value-based methods such as 𝑄-learning (Du et al., 2016; Pendharkar & Cusatis, 2018),
SARSA (Pendharkar & Cusatis, 2018), and DQN (Park et al., 2020), and policy-based algorithms
such as DPG and DDPG (Xiong et al., 2018; Jiang et al., 2017; Yu et al., 2019; Liang et al., 2018;
Aboussalah, 2020; Cong et al., 2021) have been applied to solve portfolio optimization problems.
The state variables are often composed of time, asset prices, asset past returns, current holdings of
assets, and remaining balance. The control variables are typically set to be the amount/proportion
of wealth invested in each component of the portfolio. Examples of reward signals include port-
folio return (Jiang et al., 2017; Pendharkar & Cusatis, 2018; Yu et al., 2019), (differential) Sharpe
ratio (Du et al., 2016; Pendharkar ... Cusatis, 2018), and profit (Du et al., 2016). The benchmark
strategies include Constantly Rebalanced Portfolio (CRP) (Yu et al., 2019; Jiang et al., 2017) where
at each period the portfolio is rebalanced to the initial wealth distribution among the assets, and
the buy-and-hold or do-nothing strategy (Park et al., 2020; Aboussalah, 2020), which does not take
any action but rather holds the initial portfolio until the end. The performance measures studied
in these papers include the Sharpe ratio (Yu et al., 2019; Wang ... Zhou, 2020; Xiong et al., 2018;
Jiang et al., 2017; Liang et al., 2018; Park et al., 2020; Wang, 2019), the Sortino ratio (Yu et al.,
2019), portfolio returns (Aboussalah, 2020; Liang et al., 2018; Park et al., 2020; Wang, 2019; Xiong
et al., 2018; Yu et al., 2019), portfolio values (Jiang et al., 2017; Pendharkar & Cusatis, 2018; Xiong
et al., 2018), and cumulative profits (Du et al., 2016). Some models incorporate the transaction
costs (Aboussalah, 2020; Du et al., 2016; Jiang et al., 2017; Liang et al., 2018; Park et al., 2020; Yu
et al., 2019) and investments in the risk-free asset (Du et al., 2016; Jiang et al., 2017; Wang ... Zhou,
2020; Wang, 2019; Yu et al., 2019).
For value-based algorithms, Du et al. (2016) considered the portfolio optimization problems of a
risky asset and a risk-free asset. They compared the performance of the 𝑄-learning algorithm and
a Recurrent RL (RRL) algorithm under three different value functions including the Sharpe ratio,
differential Sharpe ratio, and profit. The RRL algorithm is a policy-based method, which uses
the last action as an input. They concluded that the 𝑄-learning algorithm is more sensitive to the
choice of value function and has less stable performance than the RRL algorithm. They also sug-
gested that the (differential) Sharpe ratio is preferred rather than the profit as the reward function.
Pendharkar and Cusatis (2018) studied a two-asset personal retirement portfolio optimization
problem, in which traders who manage retirement funds are restricted from making frequent
trades and they can only access limited information. They tested the performance of three algo-
rithms: SARSA and 𝑄-learning methods with discrete state and action space that maximize either
the portfolio return or differential Sharpe ratio, and a TD learning method with discrete state
space and continuous action space that maximizes the portfolio return. The TD method learns the
portfolio returns by using a linear regression model and was shown to outperform the other two
methods in terms of portfolio values. Park et al. (2020) proposed a portfolio trading strategy based
on DQN, which chooses to either hold, buy, or sell a prespecified quantity of the asset at each time
point. In their experiments with two different three-asset portfolios, their trading strategy is supe-
rior to four benchmark strategies including the do-nothing strategy and a random strategy (take
an action in the feasible space randomly) using performance measures including the cumulative
return and the Sharpe ratio. Dixon and Halperin (2020) apply a G-learning-based algorithm,
a probabilistic extension of 𝑄-learning, which scales to high-dimensional portfolios while

providing a flexible choice of utility functions, for wealth management problems. In addition, the
authors also extend the G-learning algorithm to the setting of Inverse Reinforcement Learning
(IRL) where rewards collected by the agent are not observed, and should instead be inferred.
For policy-based algorithms, Jiang et al. (2017) proposed a framework combining neural net-
works with DPG. They used a so-called Ensemble of Identical Independent Evaluators (EIIE)
topology to predict the potential growth of the assets in the immediate future using historical data,
which includes the highest, lowest, and closing prices of portfolio components. The experiments
using real cryptocurrency market data showed that their framework achieves higher Sharpe ratio
and cumulative portfolio values compared with three benchmarks including CRP and several
published RL models. Xiong et al. (2018) explored the DDPG algorithm for the portfolio selection
of 30 stocks, where at each time point, the agent can choose to buy, sell, or hold each stock. The
DDPG algorithm was shown to outperform two classical strategies including the min-variance
portfolio allocation method (Yang et al., 2018) in terms of several performance measures includ-
ing final portfolio values, annualized return, and Sharpe ratio, using historical daily prices of the
30 stocks. Liang et al. (2018) considered the DDPG, PPO, and policy gradient method with an
adversarial learning scheme, which learns the execution strategy using noisy market data. They
applied the algorithms for portfolio optimization to five stocks using market data and the pol-
icy gradient method with adversarial learning scheme achieves higher daily return and Sharpe
ratio than the benchmark CRP strategy. They also showed that the DDPG and PPO algorithms
fail to find the optimal policy even in the training set. Yu et al. (2019) proposed a model using
DDPG, which includes prediction of the assets future price movements based on historical prices
and synthetic market data generation using a Generative Adversarial Network (GAN) (Goodfellow
et al., 2014). The model outperforms the benchmark CRP and the model considered in Jiang et al.
(2017) in terms of several performance measures including Sharpe and Sortino ratios. Cong et al.
(2021) embedded alpha portfolio strategies into a deep policy-based method and designed a frame-
work, which is easier to interpret. Using Actor–Critic methods, Aboussalah (2020) combined the
mean–variance framework (the actor determines the policy using the mean–variance framework)
and the Kelly Criterion framework (the critic evaluates the policy using their growth rate). They
studied eight policy-based algorithms including DPG, DDPG, and PPO, among which DPG was
shown to achieve the best performance.
In addition to the above discrete-time models, Wang and Zhou (2020) studied the entropy-
regularized continuous-time mean–variance framework with one risky asset and one risk-free
asset. They proved that the optimal policy is Gaussian with decaying variance and proposed an
Exploratory Mean-Variance (EMV) algorithm, which consists of three procedures: policy eval-
uation, policy improvement, and a self-correcting scheme for learning the Lagrange multiplier.
They showed that this EMV algorithm outperforms two benchmarks, including the analytical
solution with estimated model parameters obtained from the classical maximum likelihood
method (Campbell et al., 1997, Section 9.3.2) and a DDPG algorithm, in terms of annualized
sample mean and sample variance of the terminal wealth, and Sharpe ratio in their simulations.
The continuous-time framework in Wang and Zhou (2020) was then generalized in Wang (2019)
to large scale portfolio selection setting with 𝑑risky assets and one risk-free asset. The optimal
policy is shown to be multivariate Gaussian with decaying variance. They tested the performance
of the generalized EMV algorithm on price data from the stocks in the S&P 500 with 𝑑≥20 and
it outperforms several algorithms including DDPG.

## 4.4 Option pricing and hedging
Understanding how to price and hedge financial derivatives is a cornerstone of modern math-
ematical and computational finance due to its importance in the finance industry. A financial
derivative is a contract that derives its value from the performance of an underlying entity. For
example, a call or put option is a contract, which gives the holder the right, but not the obligation,
to buy or sell an underlying asset or instrument at a specified strike price prior to or on a specified
date called the expiration date. Examples of option types include European options, which can
only be exercised at expiry, and American options, which can be exercised at any time before the
option expires.
The Black–Scholes model.
One of the most important mathematical models for option pricing is the Black–Scholes or Black–
Scholes–Merton (BSM) model (Black ... Scholes, 1973; Merton, 1973), in which we aim to find the
price of a European option 𝑉(𝑆𝑡, 𝑡) (0 ≤𝑡≤𝑇) with underlying stock price 𝑆𝑡, expiration time 𝑇,
and payoff at expiry 𝑃(𝑆𝑇). The underlying stock price 𝑆𝑡is assumed to be nondividend paying
and to follow a geometric Brownian motion
𝑑𝑆𝑡= 𝜇𝑆𝑡𝑑𝑡+ 𝜎𝑆𝑡𝑑𝑊𝑡,
where 𝜇and 𝜎are called the drift and volatility parameters of the underlying asset,
and 𝑊= {𝑊𝑡}0≤𝑡≤𝑇is a standard Brownian motion defined on a filtered probability space
(Ω, , {}0≤𝑡≤𝑇, ℙ). The key idea of the BSM is that European options can be perfectly replicated
by a continuously rebalanced portfolio of the underlying asset and a riskless asset, under the
assumption that there are no market frictions, and that trading can take place continuously and in
arbitrarily small quantities. If the derivative can be replicated, by analyzing the cost of the hedg-
ing strategy, the derivative’s price must satisfy the following Black–Scholes partial differential
equation
𝜕𝑉
𝜕𝑡(𝑠, 𝑡) + 1
2𝜎2𝑠2 𝜕2𝑉
𝜕𝑠2 (𝑠, 𝑡) + 𝑟𝑠𝜕𝑉
𝜕𝑠(𝑠, 𝑡) −𝑟𝑉(𝑠, 𝑡) = 0,
(45)
with terminal condition 𝑉(𝑠, 𝑇) = 𝑃(𝑠) and where 𝑟is the known (constant) risk-free interest rate.
When we have a call option with payoff 𝑃(𝑠) = max(𝑠−𝐾, 0), giving the option buyer the right to
buy the underlying asset, the solution to the Black–Scholes equation is given by
𝑉(𝑠, 𝑡) = 𝑁(𝑑1)𝑠−𝑁(𝑑2)𝐾𝑒−𝑟(𝑇−𝑡),
where 𝑁(⋅) is the standard normal cumulative distribution function, and 𝑑1 and 𝑑2 are given by
𝑑1 =
1
𝜎
√
𝑇−𝑡
(
ln
( 𝑠
𝐾
)
+
(
𝑟+ 𝜎2
2
)
(𝑇−𝑡)
)
,
𝑑2 = 𝑑1 −𝜎
√
𝑇−𝑡.
We refer to the survey Broadie and Detemple (2004) for details about extensions of the BSM model,
other classical option pricing models, and numerical methods such as the Monte Carlo method.
In a complete market, one can hedge a given derivative contract by buying and selling the
underlying asset in the right way to eliminate risk. In the Black–Scholes analysis, delta hedging

is used, in which we hedge the risk of a call option by shorting Δ𝑡units of the underlying asset
with Δ𝑡∶= 𝜕𝑉
𝜕𝑆(𝑆𝑡, 𝑡) (the sensitivity of the option price with respect to the asset price). It is also
possible to use financial derivatives to hedge against the volatility of given positions in the under-
lying assets. However, in practice, we can only rebalance portfolios at discrete time points and
frequent transactions may incur high costs. Therefore an optimal hedging strategy depends on
the tradeoff between the hedging error and the transaction costs. It is worth mentioning that this
is in a similar spirit to the mean–variance portfolio optimization framework introduced in Sec-
tion 4.3. Much effort has been made to include the transaction costs using classical approaches
such as dynamic programming, see, for example, Leland (1985), Figlewski (1989), Henrotte (1993).
We also refer to Giurca and Borovkova (2021) for the details about delta hedging under different
model assumptions.
However, the above discussion addresses option pricing and hedging in a model-based way
since there are strong assumptions made on the asset price dynamics and the form of transaction
costs. In practice, some assumptions of the BSM model are not realistic since: (1) transaction costs
due to commissions, market impact, and nonzero bid–ask spread exist in the real market; (2) the
volatility is not constant; (3) short-term returns typically have a heavy-tailed distribution (Cont,
2001; Chakraborti et al., 2011). Thus the resulting prices and hedges may suffer from model mis-
specification when the real asset dynamics is not exactly as assumed and the transaction costs
are difficult to model. Thus, we will focus on a model-free RL approach that can address some of
these issues.
RL approach.
RL methods including (deep) 𝑄-learning (Cao et al., 2021; Du et al., 2020; Halperin, 2020;
Halperin, 2019; Li et al., 2009), PPO (Du et al., 2020), and DDPG (Cao et al., 2021) have been
applied to find hedging strategies and price financial derivatives. The state variables often include
asset price, current positions, option strikes, and time remaining to expiry. The control variable is
often set to be the change in holdings. Examples of reward functions are (risk-adjusted) expected
wealth/return (Du et al., 2020; Halperin, 2020, 2019; Kolm & Ritter, 2019) (as in the mean–variance
portfolio optimization), option payoff (Li et al., 2009), and (risk-adjusted) hedging cost (Cao et al.,
2021). The benchmarks for pricing models are typically the BSM model (Halperin, 2020, 2019)
and binomial option pricing model (Dubrov, 2015) (introduced in Cox et al. (1979)). The learned
hedging strategies are typically compared to a discrete approximation to the delta hedging strat-
egy for the BSM model (Buehler et al., 2019; Cannelli et al., 2020; Cao et al., 2021; Du et al., 2020),
or the hedging strategy for the Heston model (Buehler et al., 2019). In contrast to the BSM model
where the volatility is assumed to be constant, the Heston model assumes that the volatility of the
underlying asset follows a particular stochastic process, which leads to a semi-analytical solution.
The performance measures for the hedging strategy used in RL papers include (expected) hedg-
ing cost/error/loss (Buehler et al., 2019; Cannelli et al., 2020; Cao et al., 2021; Kolm & Ritter, 2019),
PnL (Cannelli et al., 2020; Du et al., 2020; Kolm & Ritter, 2019), and average payoff (Li et al., 2009).
Some practical issues have been taken into account in RL models, including transaction costs (Cao
et al., 2021; Buehler et al., 2019; Du et al., 2020; Kolm & Ritter, 2019) and position constraints such
as round lotting (Du et al., 2020) (where a round lot is a standard number of securities to be traded,
such as 100 shares) and limits on the trading size (for example buy or sell up to 100 shares) (Kolm
& Ritter, 2019).
For European options, Halperin (2020) developed a discrete-time option pricing model called
the QLBS (𝑄-Learner in Black–Scholes) model based on Fitted 𝑄-learning (see Section 3.2). This

model learns both the option price and the hedging strategy in a similar spirit to the mean–
variance portfolio optimization framework. Halperin (2019) extended the QLBS model in Halperin
(2020) by using Fitted 𝑄-learning. They also investigated the model in a different setting where
the agent infers the risk-aversion parameter in the reward function using observed states and
actions. However, Halperin (2020) and Halperin (2019) did not consider transaction costs in their
analysis. By contrast, Buehler et al. (2019) used deep neural networks to approximate an optimal
hedging strategy under market frictions, including transaction costs, and convex risk measures
(Klöppel ... Schweizer, 2007) such as conditional value at risk. They showed that their method
can accurately recover the optimal hedging strategy in the Heston model (Heston, 1993) without
transaction costs and it can be used to numerically study the impact of proportional transaction
costs on option prices. The learned hedging strategy is also shown to outperform delta hedging in
the (daily recalibrated) BSM model in a simple setting, this is hedging an at-the-money European
call option on the S&P 500 index. The method in Buehler et al. (2019) was extended in Carbonneau
and Godin (2021) to price and hedge a very large class of derivatives including vanilla options and
exotic options in more complex environments (e.g., stochastic volatility models and jump pro-
cesses). Kolm and Ritter (2019) found the optimal hedging strategy by optimizing a simplified
version of the mean–variance objective function (43), subject to discrete trading, round lotting,
and nonlinear transaction costs. They showed in their simulations that the learned hedging strat-
egy achieves a much lower cost, with no significant difference in volatility of total PnL, compared
to the delta hedging strategy. Du et al. (2020) extended the framework in Kolm and Ritter (2019)
and tested the performance of DQN and PPO for European options with different strikes. Their
simulation results showed that these models are superior to delta hedging in general, and out of
all models, PPO achieves the best performance in terms of PnL, training time, and the amount
of data needed for training. Cannelli et al. (2020) formulated the optimal hedging problem as a
Risk-averse Contextual 𝑘-Armed Bandit (R-CMAB) model (see the discussion of the contextual
bandit problem in Section 2.2) and proposed a deep CMAB algorithm involving Thompson Sam-
pling (Thompson, 1933). They showed that their algorithm outperforms DQN in terms of sample
efficiency and hedging error when compared to delta hedging (in the setting of the BSM model).
Their learned hedging strategy was also shown to converge to delta hedging in the absence of risk
adjustment, discretization error, and transaction costs. Cao et al. (2021) considered 𝑄-learning
and DDPG for the problem of hedging a short position in a call option when there are transaction
costs. The objective function is set to be a weighted sum of the expected hedging cost and the stan-
dard deviation of the hedging cost. They showed that their approach achieves a markedly lower
expected hedging cost but with a slightly higher standard deviation of the hedging cost when com-
pared to delta hedging. In their simulations, the stock price is assumed to follow either geometric
Brownian motion or a stochastic volatility model.
For American options, the key challenge is to find the optimal exercise strategy, which
determines when to exercise the option as this determines the price. Li et al. (2009) used the Least-
Squares Policy Iteration (LSPI) algorithm (Lagoudakis ... Parr, 2003) and the Fitted 𝑄-learning
algorithm to learn the exercise policy for American options. In their experiments for American
put options using both real and synthetic data, the two algorithms gain larger average payoffs than
the benchmark Longstaf–Schwartz method (Longstaff ... Schwartz, 2001), which is the standard
Least-Squares Monte Carlo algorithm. Li et al. (2009) also analyzed their approach from a theoret-
ical perspective and derived a high probability, finite-time bound on their method. Dubrov (2015)
then extended the work in Li et al. (2009) by combining random forests, a popular machine learn-
ing technique, with Monte Carlo simulation for pricing of both American options and convertible
bonds, which are corporate bonds that can be converted to the stock of the issuing company by

the bond holder. They showed that the proposed algorithm provides more accurate prices than
several other methods including LSPI, Fitted 𝑄-learning, and the Longstaf–Schwartz method.
## 4.5 Market making
A market maker in a financial instrument is an individual trader or an institution that provides
liquidity to the market by placing buy and sell limit orders in the LOB for that instrument while
earning the bid–ask spread.
The objective in market making is different from problems in optimal execution (to target a
position) or portfolio optimization (for long-term investing). Instead of profiting from identifying
the correct price movement direction, the objective of a market maker is to profit from earning the
bid–ask spread without accumulating undesirably large positions (known as inventory) (Guéant
et al., 2012). A market maker faces three major sources of risk (Guilbaud ... Pham, 2013). The
inventory risk (Avellaneda ... Stoikov, 2008) refers to the risk of accumulating an undesirable
large net inventory, which significantly increases volatility due to market movements. The exe-
cution risk (Kühn ... Stroh, 2010) is the risk that limit orders may not get filled over a desired
horizon. Finally, the adverse selection risk refers to the situation where there is a directional price
movement that sweeps through the limit orders submitted by the market marker such that the
price does not revert back by the end of the trading horizon. This may lead to a huge loss as the
market maker in general needs to clear their inventory at the end of the horizon (typically the end
of the day to avoid overnight inventory).
Stochastic control approach.
Traditionally, the theoretical study of market making strategies follows a stochastic control
approach, where the LOB dynamics are modeled directly by some stochastic process and an opti-
mal market making strategy that maximizes the market maker’s expected utility can be obtained
by solving the Hamilton–Jacobi–Bellman equation. See Avellaneda and Stoikov (2008), Guéant
et al. (2013), and Obizhaeva & Wang (2013), and (Cartea et al., 2015, Chapter 10) for examples.
We follow the framework in Avellaneda and Stoikov (2008) as an example to demonstrate the
control formulation in continuous time and discuss its advantages and disadvantages.
Consider a high-frequency market maker trading on a single stock over a finite horizon 𝑇.
Suppose the mid-price of this stock follows an arithmetic Brownian motion in that
𝑑𝑆𝑡= 𝜎𝑑𝑊𝑡,
(46)
where 𝜎is a constant and 𝑊= {𝑊𝑡}0≤𝑡≤𝑇is a standard Brownian motion defined on a filtered
probability space (Ω, , {}0≤𝑡≤𝑇, ℙ).
The market maker will continuously propose bid and ask prices, denoted by 𝑆𝑏
𝑡and 𝑆𝑎
𝑡, respec-
tively, and hence will buy and sell shares according to the rate of arrival of market orders at the
quoted prices. Her inventory, which is the number of shares she holds, is, therefore, given by
𝑞𝑡= 𝑁𝑏
𝑡−𝑁𝑎
𝑡,
(47)
where 𝑁𝑏and 𝑁𝑎are the point processes (independent of 𝑊) giving the number of shares the
market maker, respectively, bought and sold. Arrival rates obviously depend on the prices 𝑆𝑏
𝑡and
𝑆𝑎
𝑡quoted by the market maker and we assume that the intensities 𝜆𝑏and 𝜆𝑎associated, respec-

tively, to 𝑁𝑏and 𝑁𝑎depend on the difference between the quoted prices and the reference price
(i.e., 𝛿𝑏
𝑡= 𝑆𝑡−𝑆𝑏
𝑡and 𝛿𝑎
𝑡= 𝑆𝑡−𝑆𝑎
𝑡) and are of the following form:
𝜆𝑏(𝛿𝑏) = 𝐴exp(−𝑘𝛿𝑏) and 𝜆𝑎(𝛿𝑎) = 𝐴exp(−𝑘𝛿𝑎),
(48)
where 𝐴and 𝑘are positive constants that characterize the liquidity of the stock. Consequently,
the cash process of the market maker follows:
𝑑𝑋𝑡= (𝑆𝑡+ 𝛿𝑎
𝑡)𝑑𝑁𝑎
𝑡−(𝑆𝑡−𝛿𝑏
𝑡)𝑑𝑁𝑏
𝑡.
(49)
Finally, the market maker optimizes a constant absolute risk aversion (CARA) utility function:
𝑉(𝑠, 𝑥, 𝑞, 𝑡) =
sup
{𝛿𝑎
𝑢,𝛿𝑏
𝑢}𝑡≤𝑢≤𝑇∈
𝔼
[
−exp(−𝛾(𝑋𝑇+ 𝑞𝑇𝑆𝑇))
||||
𝑋𝑡= 𝑥, 𝑆𝑡= 𝑠, and 𝑞𝑡= 𝑞
]
,
(50)
where is the set of predictable processes bounded from below, 𝛾is the absolute risk aversion
coefficient characterizing the market maker, 𝑋𝑇is the amount of cash at time 𝑇and 𝑞𝑇𝑆𝑇is the
evaluation of the (signed) remaining quantity of shares in the inventory at time 𝑇.
By applying the DPP, the value function 𝑉solves the following Hamilton–Jacobi–Bellman
equation:
⎧
⎪
⎨
⎪⎩
𝜕𝑡𝑉+ 1
2𝜕𝑠𝑠𝑉+ max𝛿𝑏𝜆𝑏(𝛿𝑏)[𝑉(𝑠, 𝑥−𝑠+ 𝛿𝑏, 𝑞+ 1, 𝑡) −𝑢(𝑠, 𝑥, 𝑞, 𝑡)]
+ max𝛿𝑎𝜆𝑎(𝛿𝑎)[𝑉(𝑠, 𝑥+ 𝑠+ 𝛿𝑎, 𝑞−1, 𝑡) −𝑢(𝑠, 𝑥, 𝑞, 𝑡)] = 0,
𝑉(𝑠, 𝑥, 𝑞, 𝑇) = −exp(−𝛾(𝑥+ 𝑞𝑠)).
(51)
While Equation (51), derived in Avellaneda and Stoikov (2008), admits a (semi) closed-form
solution, which leads to nice insights about the problem, it all builds on the full analytical specifi-
cation of the market dynamics. In addition, there are very few utility functions (e.g., exponential
(CARA), power (CRRA), and quadratic) known in the literature that could possibly lead to closed-
form solutions. The same issues arise in other work along this line (Guéant et al., 2013; Obizhaeva
& Wang, 2013; Cartea et al., 2015, Chapter 10) where strong model assumptions are made about
the prices or about the LOB or both. This requirement of full analytical specification means these
papers are quite removed from realistic market making, as financial markets do not conform to
any simple parametric model specification with fixed parameters.
RL approach.
For market making problems with an RL approach, both value-based methods (such as the 𝑄-
learning algorithm (Abernethy & Kale, 2013; Spooner et al., 2018) and SARSA (Spooner et al.,
2018)) and policy-based methods (such as deep policy gradient method Zhao ... Linetsky, 2021)
have been used. The state variables are often composed of bid and ask prices, current holdings
of assets, order-flow imbalance, volatility, and some sophisticated market indices. The control
variables are typically set to be the spread to post a pair of limit buy and limit sell orders. Examples
of reward include PnL with inventory cost (Abernethy & Kale, 2013; Spooner et al., 2018; Spooner
& Savani, 2020; Zhao & Linetsky, 2021) or Implementation Shortfall with inventory cost (Gašperov
and Kostanjčar, 2021).

The first RL method for market making was explored by Chan and Shelton (2001) and the
authors applied three RL algorithms and tested them in simulation environments: a Monte Carlo
method, a SARSA method, and an Actor–Critic method. For all three methods, the state variables
include inventory of the market-maker, order imbalance on the market, and market quality mea-
sures. The actions are the changes in the bid/ask price to post the limit orders and the sizes of
the limit buy/sell orders. The reward function is set as a linear combination of profit (to maxi-
mize), inventory risk (to minimize), and market qualities (to maximize). The authors found that
SARSA and Monte Carlo methods work well in a simple simulation environment. The Actor–
Critic method is more plausible in complex environments and generates stochastic policies that
correctly adjust bid/ask prices with respect to order imbalance and effectively control the trade-off
between the profit and the quoted spread (defined as the price difference to post limit buy orders
and limit sell orders). Furthermore, the stochastic policies are shown to outperform deterministic
policies in achieving a lower variance of the resulting spread. Later on, Abernethy and Kale (2013)
designed a group of “spread-based” market making strategies parametrized by a minimum quoted
spread. The strategies bet on the mean-reverting behavior of the mid-price and utilize the oppor-
tunities when the mid-price deviates from the price during the previous period. Then an online
algorithm is used to pick in each period a minimum quoted spread. The states of the market maker
are the current inventory and price data. The actions are the quantities and at what prices to offer
in the market. It is assumed that the market maker interacts with a continuous double auction
via an order book. The market maker can place both market and limit orders and is able to make
and cancel orders after every price fluctuation. The authors provided structural properties of these
strategies, which allows them to obtain low regret relative to the best such strategy in hindsight,
which maximizes the realized rewards. Spooner et al. (2018) generalized the results in Chan and
Shelton (2001) and adopted several RL algorithms (including 𝑄-learning and SARSA) to improve
the decisions of the market maker. In their framework, the action space contains 10 actions. The
first nine actions correspond to a pair of orders with a particular spread and bias in their prices.
The final action allows the agent to clear their inventory using a market order. The states can
be divided into two groups: agent states and market states. Agent states include inventory level
and active quoting distances and market states include market (bid–ask) spread, mid-price move,
book/queue imbalance, signed volume, volatility, and relative strength index. The reward function
is designed as the sum of a symmetrically dampened PnL and an inventory cost. The idea of the
symmetric damping is to disincentivize the agent from trend chasing and direct the agent towards
spread capturing. A simulator of a financial market is constructed via direct reconstruction of the
LOB from historical data. Although, since the market is reconstructed from historical data, simu-
lated orders placed by an agent cannot impact the market. The authors compared their algorithm
to a modified version of Abernethy and Kale (2013) and showed a significant empirical improve-
ment from Abernethy and Kale (2013). To address the adverse selection risk that market makers
are often faced with in a high-frequency environment, Zhao and Linetsky (2021) proposes a high-
frequency feature book exhaustion rate (BER) and shows theoretically and empirically that the
BER can serve as a direct measurement of the adverse selection risk from an equilibrium point of
view. The authors train a market making algorithm via a deep policy-based RL algorithm using 3
years of LOB data for the Chicago Mercantile Exchange (CME) S&P 500 and 10-year Treasury note
futures. The numerical performance demonstrates that utilizing the BER allows the algorithm to
avoid large losses due to adverse selection and achieve stable performance.
Some recent papers have focused on improving the robustness of market markers’ strate-
gies with respect to adversarial and volatile market conditions. Gašperov and Kostanjčar (2021)
used perturbations by an opposing agent—the adversary—to render the market maker more

robust to model uncertainty and consequently improve generalization. Meanwhile, Gašperov and
Kostanjčar (2021) incorporated additional predictive signals in the states to improve the learning
outcomes of market makers. In particular, the states include the agent’s inventory, price range,
and trend predictions. In a similar way to the setting in Abernethy and Kale (2013) and Spooner
et al. (2018), actions are represented by a certain choice of bid/ask orders relative to the current
best bid/ask. The reward function both incentivizes spread-capturing (making round-trips) and
discourages holding inventory. The first part of the reward is inspired by utility functions contain-
ing a running inventory-based penalty with an absolute value inventory penalty term which has a
convenient value at risk (VAR) interpretation. The second part is a symmetrically dampened PnL,
which follows Spooner et al. (2018). Experimental results on historical data demonstrate the supe-
rior reward-to-risk performance of the proposed framework over several standard market making
benchmarks. More specifically, in these experiments, the resulting RL agent achieves between
20%–30% higher terminal wealth than the benchmarks while being exposed to only around 60% of
their inventory risks. Also, Spooner and Savani (2020) applied Adversarial Reinforcement Learn-
ing (ARL) to a zero-sum game version of the control formulation (46)–(51). The adversary acts as
a proxy for other market participants that would like to profit at the market maker’s expense.
In addition to designing learning algorithms for the market maker in an unknown financial
environment, RL algorithms can also be used to solve high-dimensional control problems for the
market maker or to solve the control problem with the presence of a time-dependent rebate in the
full information setting. In particular, Guéant and Manziuk (2019) focused on a setting where a
market maker needs to decide the optimal bid and ask quotes for a given universe of bonds in an
OTC market. This problem is high-dimensional and other classical numerical methods including
finite differences are inapplicable. The authors proposed a model-based Actor–Critic-like algo-
rithm involving a deep neural network to numerically solve the problem. Similar ideas have been
applied to market making problems in dark pools (Baldacci et al., 2019) (see the discussion on
dark pools in Section 4.7). With the presence of a time-dependent rebate, there is no closed-form
solution for the associated stochastic control problem of the market maker. Instead, Zhang and
Chen (2020) proposed a Hamiltonian-guided value function approximation algorithm to solve for
the numerical solutions under this scenario.
Multi-agent RL algorithms are also used to improve the strategy for market making with a
particular focus on the impact of competition from other market makers or the interaction with
other types of market participant. See Ganesh et al. (2019) and Patel (2018).
## 4.6 Robo-advising
Robo-advisors, or automated investment managers, are a class of financial advisers that provide
online financial advice or investment management with minimal human intervention. They pro-
vide digital financial advice based on mathematical rules or algorithms, which can easily take into
account different sources of data such as news, social media information, sentiment data, and
earnings reports. Robo-advisors have gained widespread popularity and emerged prominently as
an alternative to traditional human advisers in recent years. The first robo-advisors were launched
after the 2008 financial crisis when financial services institutions were facing the ensuing loss
of trust from their clients. Examples of pioneering robo-advising firms include Betterment and
Wealthfront. As of 2020, the value of assets under robo-management is highest in the United
States and exceeded $650 billion (Capponi et al., 2021).

The robo-advisor does not know the client’s risk preference in advance but learns it while
interacting with the client. The robo-advisor then improves its investment decisions based on
its current estimate of the client’s risk preference. There are several challenges in the applica-
tion of robo-advising. First, the client’s risk preference may change over-time and may depend on
the market returns and economic conditions. Therefore, the robo-advisor needs to determine a
frequency of interaction with the client that ensures a high level of consistency in the risk prefer-
ence when adjusting portfolio allocations. Second, the robo-advisor usually faces a dilemma when
it comes to either catering to the client’s wishes, that is, investing according to the client’s risk
preference, or going against the client’s wishes in order to seek better investment performance.
Finally, there is also a subtle trade-off between the rate of information acquisition from the client
and the accuracy of the acquired information. On the one hand, if the interaction does not occur
at all times, the robo-advisor may not always have access to up-to-date information about the
client’s profile. On the other hand, information communicated to the robo-advisor may not be
representative of the client’s true risk aversion as the client is subject to behavioral biases.
Stochastic control approach.
To address the above-mentioned challenges, Capponi et al. (2021) proposed a stochastic control
framework with four components: (i) a regime switching model of market returns, (ii) a mech-
anism of interaction between the client and the robo-advisor, (iii) a dynamic model (i.e., risk
aversion process) for the client’s risk preferences, and (vi) an optimal investment criterion. In
this framework, the robo-advisor interacts repeatedly with the client and learns about changes in
her risk profile whose evolution is specified by (iii). The robo-advisor adopts a multiperiod mean–
variance investment criterion with a finite investment horizon based on the estimate of the client’s
risk aversion level. The authors showed the stock market allocation resulting from the dynamic
mean–variance optimization consists of two components where the first component is akin to the
standard single period Markowitz strategy whereas the second component is intertemporal hedg-
ing demand, which depends on the relationship between the current market return and future
portfolio returns.
Note that although Capponi et al. (2021) focused on the stochastic control approach to tackle
the robo-advising problem, the framework is general enough that some components (for example,
the mean–variance optimization step) may be replaced by an RL algorithm and the dependence
on model specification can be potentially relaxed.
RL approach.
There are only a few references on robo-advising with an RL approach since this is still a rela-
tively new topic. We review each paper with details. The first RL algorithm for a robo-advisor
was proposed by Alsabah et al. (2021) where the authors designed an exploration–exploitation
algorithm to learn the investor’s risk appetite over time by observing her portfolio choices in dif-
ferent market environments. The set of various market environments of interest is formulated
as the state space . In each period, the robo-advisor places an investor’s capital into one of
several pre-constructed portfolios, which can be viewed as the action space . Each portfolio
decision reflects the robo-advisor’s belief concerning the investor’s true risk preference from a
discrete set of possible risk aversion parameters Θ = {𝜃𝑖}1≤𝑖≤|Θ|. The investor interacts with the
robo-advisor by portfolio selection choices, and such interactions are used to update the robo-
advisor’s estimate of the investor’s risk profile. The authors proved that, with high probability, the
proposed exploration–exploitation algorithm performs near optimally with the number of time
steps depending polynomially on various model parameters.

Wang and Yu (2021) proposed an investment robo-advising framework consisting of two agents.
The first agent, an inverse portfolio optimization agent, infers an investor’s risk preference and
expected return directly from historical allocation data using online inverse optimization. The
second agent, a deep RL agent, aggregates the inferred sequence of expected returns to formulate
a new multiperiod mean–variance portfolio optimization problem that can be solved using a deep
RL approach based on the DDPG method. The proposed investment pipeline was applied to real
market data from April 1, 2016 to February 1, 2021 and was shown to consistently outperform the
S&P 500 benchmark portfolio that represents the aggregate market optimal allocation.
As mentioned earlier in this subsection, learning the client’s risk preference is challenging as
the preference may depend on multiple factors and may change over time. Yu et al. (2020) was ded-
icated to learning the risk preferences from investment portfolios using an inverse optimization
technique. In particular, the proposed inverse optimization approach can be used to measure time-
varying risk preferences directly from market signals and portfolios. This approach is developed
based on two methodologies: convex optimization-based modern portfolio theory and learning
the decision-making scheme through inverse optimization.
## 4.7 Smart order routing
In order to execute a trade of a given asset, market participants may have the opportunity to split
the trade and submit orders to different venues, including both lit pools and dark pools, where
this asset is traded. This could potentially improve the overall execution price and quantity. Both
the decision and hence the outcome are influenced by the characteristics of different venues as
well as the structure of transaction fees and rebates across different venues.
Dark pools versus lit pools.
Dark pools are private exchanges for trading securities that are not accessible by the investing
public. Also known as “dark pools of liquidity,” the name of these exchanges is a reference to
their complete lack of transparency. Dark pools were created in order to facilitate block trading by
institutional investors who did not wish to impact the markets with their large orders and obtain
adverse prices for their trades. According to recent Securities and Exchange Commission (SEC)
data, there were 59 registered Alternative Trading Systems (a.k.a. the “Dark Pools”) with the SEC
as of May 2021 of which there are three types: (1) Broker-Dealer-Owned Dark Pools, (2) Agency
Broker or Exchange-Owned Dark Pools, and (3) Electronic Market Makers Dark Pools. Lit pools
are effectively the opposite of dark pools. Unlike dark pools, where prices at which participants
are willing to trade are not revealed, lit pools do display bid offers and ask offers in different stocks.
Primary exchanges operate in such a way that available liquidity is displayed at all times and form
the bulk of the lit pools available to traders.
For smart order routing (SOR) problems, the most important characteristics of different dark
pools are the chances of being matched with a counterparty and the price (dis)advantages whereas
the relevant characteristics of lit pools include the order flows, queue sizes, and cancellation rates.
There are only a few references on using a data-driven approach to tackle SOR problems for
dark pool allocations and for allocations across lit pools. We will review each of them with details.

Allocation across lit pools.
The SOR problem across multiple lit pools (or primary exchanges) was first studied in Cont and
Kukanov (2017) where the authors formulated the SOR problem as a convex optimization prob-
lem.
Consider a trader who needs to buy 𝑆shares of a stock within a short time interval [0, 𝑇]. At
time 0, the trader may submit 𝐾limit orders with 𝐿𝑘≥0 shares to exchanges 𝑘= 1, … , 𝐾(join-
ing the queue of the best bid price level) and also market orders for 𝑀≥0 shares. At time 𝑇
if the total executed quantity is less than 𝑆, the trader also submits a market order to execute
the remaining amount. The trader’s order placement decision is thus summarized by a vector
𝑋= (𝑀, 𝐿1, … , 𝐿𝐾) ∈ℝ𝐾+1
+
of order sizes. It is assumed for simplicity that a market order of any
size up to 𝑆can be filled immediately at any single exchange. Thus a trader chooses the cheapest
venue for his market orders.
Limit orders with quantities (𝐿1, … , 𝐿𝐾) join queues of (𝑄1, … , 𝑄𝐾) orders in 𝐾LOBs, where
𝑄𝑘≥0. Then the filled amounts at the end of the horizon 𝑇can be written as a function of their
initial queue position and future order flow:
min(max(𝜉𝑘−𝑄𝑘, 0), 𝐿𝑘) = (𝜉𝑘−𝑄𝑘)+ −(𝜉𝑘−𝑄𝑘−𝐿𝑘)+
(52)
where the notation (𝑥)+ = max(0, 𝑥). Here 𝜉𝑘∶= 𝐷𝑘+ 𝐶𝑘is the total outflow from the front of
the 𝑘th queue, which consists of order cancellations 𝐶𝑘that occurred before time 𝑇from queue
positions in front of an order and market orders 𝐷𝑘that reach the 𝑘th exchange before 𝑇. Note that
the fulfilled amounts are random because they depend on queue outflows 𝜉= (𝜉1, … , 𝜉𝐾) during
[0, 𝑇], which are modeled as random variables with a distribution 𝐹.
Using the mid-quote price as a benchmark, the execution cost relative to the mid-quote for an
order allocation 𝑋= (𝑀, 𝐿1, ..., 𝐿𝐾) is defined as
𝑉execution(𝑋, 𝜉) ∶= (ℎ+ 𝑓)𝑀−
𝐾
∑
𝑘=1
(ℎ+ 𝑟𝑘)((𝜉𝑘−𝑄𝑘)+ −(𝜉𝑘−𝑄𝑘−𝐿𝑘)+),
(53)
where ℎis one-half of the bid–ask spread at time 0, 𝑓is a fee for market orders and 𝑟𝑘are effective
rebates for providing liquidity by submitting limit orders on exchanges 𝑘= 1, … , 𝐾.
Penalties for violations of the target quantity in both directions are included:
𝑉penalty(𝑋, 𝜉) ∶= 𝜆𝑢(𝑆−𝐴(𝑋, 𝜉))+ + 𝜆𝑜(𝐴(𝑋, 𝜉) −𝑆)+,
(54)
where 𝜆𝑜≥0 and 𝜆𝑢≥0 are, respectively, the penalty for overshooting and undershooting, and
𝐴(𝑋, 𝜉) = 𝑀+ ∑𝐾
𝑘=1(ℎ+ 𝑟𝑘)((𝜉𝑘−𝑄𝑘)+ −(𝜉𝑘−𝑄𝑘−𝐿𝑘)+) is the total number of shares bought
by the trader during [0, 𝑇).
The impact cost is paid on all orders placed at times 0 and 𝑇, irrespective of whether they are
filled, leading to the following total impact:
𝑉impact(𝑋, 𝜉) = 𝜃(𝑀+ 𝐿𝑘+ (𝑆−𝐴(𝑋, 𝜉)+))
(55)
where 𝜃> 0 is the impact coefficient.
Finally, the cost function is defined as the summation of all three pieces 𝑉(𝑋, 𝜉) ∶=
𝑉execution(𝑋, 𝜉) + 𝑉penalty(𝑋, 𝜉) + 𝑉impact(𝑋, 𝜉). Cont and Kukanov (2017, Proposition 4) provides

optimality conditions for an order allocation 𝑋∗= (𝑀∗, 𝐿1, … , 𝐿𝐾). In particular, (semi)-explicit
model conditions are given for when 𝐿∗
𝑘> 0 (𝑀∗> 0), that is, when submitting limit orders to
venue 𝑘(when submitting market orders) is optimal.
In a different approach to the single-period model introduced above, Baldacci and Manziuk
(2020) formulated the SOR problem as an order allocation problem across multiple lit pools over
multiple trading periods. Each venue is characterized by a bid–ask spread process and an imbal-
ance process. The dependencies between the imbalance and spread at the venues are considered
through a covariance matrix. A Bayesian learning framework for learning and updating the model
parameters is proposed to take into account possibly changing market conditions. Extensions to
include short/long trading signals, market impact, or hidden liquidity are also discussed.
Allocation across dark pools.
As discussed at the beginning of Section 4.7, dark pools are a type of stock exchange that is
designed to facilitate large transactions. A key aspect of dark pools is the censored feedback that
the trader receives. At every iteration, the trader has a certain number 𝑉𝑡of shares to allocate
amongst 𝐾different dark pools with 𝑣𝑖
𝑡the volume allocated to dark pool 𝑖. The dark pool 𝑖trades
as many of the allocated shares 𝑣𝑖
𝑡as it can with the available liquidity 𝑠𝑖
𝑡. The trader only finds out
how many of these allocated shares were successfully traded at each dark pool (i.e., min(𝑠𝑖
𝑡, 𝑣𝑖
𝑡)),
but not how many would have been traded if more were allocated (i.e., 𝑠𝑖
𝑡).
Based on this property of censored feedback, Ganchev et al. (2010) formulated the allocation
problem across dark pools as an online learning problem under the assumption that 𝑠𝑖
𝑡is an i.i.d.
sample from some unknown distribution 𝑃𝑖and the total allocation quantity 𝑉𝑡is an i.i.d. sample
from an unknown distribution 𝑄with 𝑉𝑡upper bounded by a constant 𝑉> 0 almost surely. At
each iteration 𝑡, the learner allocates the orders greedily according to the estimate ˆ𝑃(𝑡−1)
𝑖
of the
distribution 𝑃𝑖for all dark pools 𝑖= 1, 2, … , 𝐾derived from previous iteration 𝑡−1. Then the
learner can update the estimation ˆ𝑃(𝑡)
𝑖
of the distribution 𝑃𝑖with a modified version of the Kaplan–
Meier estimate (a nonparametric statistic used to estimate the cumulative probability) with the
new censored observation min(𝑠𝑖
𝑡, 𝑣𝑖
𝑡) from iteration 𝑡. The authors then proved that for any 𝜀> 0
and 𝛿> 0, with probability 1 −𝛿(over the randomness of draws from Q and {𝑃𝑖}𝐾
𝑖=1), after running
for a time polynomial in 𝐾, 𝑉, 1∕𝜀, and ln(1∕𝛿), the algorithm makes an 𝜀-optimal allocation on
each subsequent time step with probability at least 1 −𝜀.
The setup of Ganchev et al. (2010) was generalized in Agarwal et al. (2010) where the authors
assumed that the sequences of volumes 𝑉𝑡and available liquidities {𝑠𝑖
𝑡}𝐾
𝑖=1 are chosen by an adver-
sary who knows the previous allocations of their algorithm. An exponentiated gradient style
algorithm was proposed and shown to enjoy an optimal regret guarantee (𝑉
√
𝑇ln 𝐾) against
the best allocation strategy in hindsight.
## 5 Further Developments For Mathematical Finance
AND REINFORCEMENT LEARNING
The general RL algorithms developed in the machine learning literature are good starting points
for use in financial applications. A possible drawback is that such general RL algorithms tend
to overfit, using more information than is actually required for a particular application. On the
other hand, the stochastic control approach to many financial decision-making problems may
suffer from the risk of model mis-specification. However, it may capture the essential features

of a given financial application from a modeling perspective, in terms of the dynamics and the
reward function.
One promising direction for RL in finance is to develop an even closer integration of the model-
ing techniques (the domain knowledge) from the stochastic control literature and key components
of a given financial application (for example the adverse selection risk for market-making prob-
lems and the execution risk for optimal liquidation problems) with the learning power of the RL
algorithms. This line of developing a more integrated framework is interesting from both theoret-
ical and applications perspectives. From the application point of view, a modified RL algorithm,
with designs tailored to one particular financial application, could lead to better empirical perfor-
mance. This could be verified by comparison with existing algorithms on the available datasets.
In addition, financial applications motivate potential new frameworks and playgrounds for RL
algorithms. Carrying out the convergence and sample complexity analysis for these modified algo-
rithms would also be a meaningful direction in which to proceed. Many of the papers referenced
in this review provide great initial steps in this direction. We list the following future directions
that the reader may find interesting.
Risk-aware or risk-sensitive RL.
Risk arises from the uncertainties associated with future events, and is inevitable since the con-
sequences of actions are uncertain at the time when a decision is made. Many decision-making
problems in finance lead to trading strategies and it is important to account for the risk of the pro-
posed strategies (which could be measured for instance by the maximum draw-down, the variance
or the 5% percentile of the PnL distribution) and/or the risk from the market environment such
as the adverse selection risk.
Hence, it would be interesting to include risk measures in the design of RL algorithms for finan-
cial applications. The challenge of risk-sensitive RL lies both in the nonlinearity of the objective
function with respect to the reward and in designing a risk-aware exploration mechanism.
RL with risk-sensitive utility functions has been studied in several papers without regard to
specific financial applications. The work of Mihatsch and Neuneier (2002) proposes TD(0) and
𝑄-learning-style algorithms that transform temporal differences instead of cumulative rewards,
and proves their convergence. Risk-sensitive RL with a general family of utility functions is stud-
ied in Shen et al. (2014), which also proposes a 𝑄-learning algorithm with convergence guarantees.
The work of Eriksson and Dimitrakakis (2019) studies a risk-sensitive policy gradient algorithm,
though with no theoretical guarantees. Fei et al. (2020) considers the problem of risk-sensitive RL
with exponential utility and proposes two efficient model-free algorithms, Risk-sensitive Value
Iteration (RSVI) and Risk-sensitive 𝑄-learning (RSQ), with a near-optimal sample complexity
guarantee. Vadori et al. (2020) developed a martingale approach to learn policies that are sensitive
to the uncertainty of the rewards and are meaningful under some market scenarios. Another line
of work focuses on constrained RL problems with different risk criteria (Achiam et al., 2017; Chow
et al., 2017, 2015; Ding et al., 2021; Tamar et al., 2015; Zheng & Ratliff, 2020). Very recently, Jaimun-
gal et al. (2021) proposed a robust risk-aware RL framework via robust optimization and with a
rank-dependent expected utility function. Financial applications such as statistical arbitrage and
portfolio optimization are discussed with detailed numerical examples. Coache and Jaimungal
(2021) develops a framework combining policy-gradient-based RL method and dynamic convex
risk measures for solving time-consistent risk-sensitive stochastic optimization problems. How-
ever, there is no sample complexity or asymptotic convergence studied for the proposed algorithms
in Jaimungal et al. (2021); Coache and Jaimungal (2021).

Offline learning and online exploration.
Online learning requires updating of algorithm parameters in real-time and this is impractical
for many financial decision-making problems, especially in the high-frequency regime. The most
plausible setting is to collect data with a prespecified exploration scheme during trading hours
and update the algorithm with the new collected data after the close of trading. This is closely
related to the translation of online learning to offline regression (Simchi-Levi & Xu, 2020) and RL
with batch data (Chen & Jiang, 2019; Gao et al., 2019; Garcelon et al., 2020; Ren & Zhou, 2020).
However, these developments focus on general methodologies without being specifically tailored
to financial applications.
Learning with a limited exploration budget.
Exploration can help agents to find new policies to improve their future cumulative rewards. How-
ever, too much exploration can be both time consuming and computation consuming, and in
particular, it may be very costly for some financial applications. Additionally, exploring black-box
trading strategies may need a lot of justification within a financial institution and hence investors
tend to limit the effort put into exploration and try to improve performance as much as possible
within a given budget for exploration. This idea is similar in spirit to conservative RL where agents
explore new strategies to maximize revenue whilst simultaneously maintaining their revenue
above a fixed baseline, uniformly over time (Wu et al., 2016). This is also related to the problem of
information acquisition with a cost which has been studied for economic commodities (Pomatto
et al., 2018) and operations management (Ke et al., 2016). It may also be interesting to investigate
such costs for decision-making problems in financial markets.
Learning with multiple objectives.
In finance, a common problem is to choose a portfolio when there are two conflicting objectives—
the desire to have the expected value of portfolio returns be as high as possible, and the desire to
have risk, often measured by the standard deviation of portfolio returns, be as low as possible.
This problem is often represented by a graph in which the efficient frontier shows the best combi-
nations of risk and expected return that are available, and in which indifference curves show the
investor’s preferences for various risk-expected return combinations. Decision makers sometimes
combine both criteria into a single objective function consisting of the difference of the expected
reward and a scalar multiple of the risk. However, it may well not be in the best interest of a deci-
sion maker to combine relevant criteria in a linear format for certain applications. For example,
market makers on the OTC market tend to view criteria such as turn around time, balance sheet
constraints, inventory cost, profit and loss as separate objective functions. The study of multi-
objective RL is still at a preliminary stage and relevant references include Zhou et al. (2020) and
Yang et al. (2019).
Learning to allocate across lit pools and dark pools.
Online optimization methods explored in Agarwal et al. (2010) and Ganchev et al. (2010) for
dark pool allocations can be viewed as a single-period RL algorithm and the Bayesian framework
developed in Baldacci and Manziuk (2020) for allocations across lit pools may be classified as a
model-based RL approach. However, there is currently no existing work on applying multiperiod
and model-free RL methods to learn how to route orders across both dark pools and lit pools. We
think this might be an interesting direction to explore as agents sometimes have access to both lit

pools and dark pools and these two contrasting pools have quite different information structures
and matching mechanisms.
Robo-advising in a model-free setting.
As introduced in Section 4.6, Alsabah et al. (2021) considered learning within a set of 𝑚
prespecified investment portfolios, and Wang and Yu (2021) and Yu et al. (2020) developed learn-
ing algorithms and procedures to infer risk preferences, respectively, under the framework of
Markowitz mean–variance portfolio optimization. It would be interesting to consider a model-
free RL approach where the robo-advisor has the freedom to learn and improve decisions beyond
a prespecified set of strategies or the Markowitz framework.
Sample efficiency in learning trading strategies.
In recent years, sample complexity has been studied extensively to understand modern RL algo-
rithms (see Sections 2 and 3.2). However, most RL algorithms still require a large number of
samples to train a decent trading algorithm, which may exceed the amount of relevant available
historical data. Financial time series are known to be nonstationary (Huang et al., 2003), and
hence historical data that are further away in time may not be helpful in training efficient learning
algorithms for the current market environment. This leads to important questions of design-
ing more sample-efficient RL algorithms for financial applications or developing good market
simulators that could generate (unlimited) realistic market scenarios Wiese et al. (2020).
Transfer learning and cold start for learning new assets.
Financial institutions or individuals may change their baskets of assets to trade over time. Possi-
ble reasons may be that new assets (for example, cooperative bonds) are issued from time to time
or the investors may switch their interest from one sector to another. There are two interesting
research directions related to this situation. When an investor has a good trading strategy, trained
by an RL algorithm for one asset, how should they transfer the experience to train a trading algo-
rithm for a “similar” asset with fewer samples? This is closely related to transfer learning (Torrey
& Shavlik, 2010; Pan & Yang, 2009). To the best of our knowledge, no study for financial appli-
cations has been carried out along this direction. Another question is the cold-start problem for
newly issued assets. When we have very limited data for a new asset, how should we initialize an
RL algorithm and learn a decent strategy using the limited available data and our experience (i.e.,
the trained RL algorithm or data) with other longstanding assets?
AC K N OW L E D G M E N T S
We thank Xuefeng Gao, Anran Hu, Xiao-Yang Liu, Wenpin Tang, Ziyi Xia, Zhuoran Yang,
Junzi Zhang, and Zeyu Zheng for helpful discussions and comments on this survey. Supported
by the EPSRC Centre for Doctoral Training in Industrially Focused Mathematical Modeling
(EP/L015803/1) in collaboration with BP plc.
O RC I D
RenyuanXu
https://orcid.org/0000-0003-4293-3450
R E F E R E N C E S
Abbasi-Yadkori, Y., Bartlett, P., Bhatia, K., Lazic, N., Szepesvari, C., & Weisz, G. (2019). Politex: Regret bounds for
policy iteration using expert prediction. In International conference on machine learning (pp. 3692–3702). PMLR.
Abernethy, J. D., & Kale, S. (2013). Adaptive market making via online learning. In NIPS (pp. 2058–2066). Citeseer.

Aboussalah, A. M. (2020). What is the value of the cross-sectional approach to deep reinforcement learning?
Available at SSRN, 22(6), 1091–1111.
Achiam, J., Held, D., Tamar, A., & Abbeel, P. (2017). Constrained policy optimization. In International conference
on machine learning (pp. 22–31). PMLR.
Agarwal, A., Bartlett, P., & Dama, M. (2010). Optimal allocation strategies for the dark pool problem. In Proceedings
of the thirteenth international conference on artificial intelligence and statistics (pp. 9–16). JMLR Workshop and
Conference Proceedings.
Agarwal, A., Kakade, S., & Yang, L. F. (2020). Model-based reinforcement learning with a generative model is
minimax optimal. In Conference on learning theory (pp. 67–83). PMLR.
Agarwal, A., Kakade, S. M., Lee, J. D., & Mahajan, G. (2021). On the theory of policy gradient methods: Optimality,
approximation, and distribution shift. Journal of Machine Learning Research, 22(98), 1–76.
Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. Journal of Risk, 3, 5–40.
Alsabah, H., Capponi, A., Ruiz Lacedelli, O., & Stern, M. (2021). Robo-advising: Learning investors’ risk preferences
via portfolio choices. Journal of Financial Econometrics, 19(2), 369–392.
Asadi, K., & Littman, M. L. (2017). An alternative softmax operator for reinforcement learning. In International
conference on machine learning (pp. 243–252). PMLR.
Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. Quantitative Finance, 8(3),
217–224.
Azar, M. G., Munos, R., & Kappen, B. (2012). On the sample complexity of reinforcement learning with a generative
model. arXiv preprint arXiv:1206.6461, 1707–1714.
Azar, M. G., Munos, R., & Kappen, H. J. (2013). Minimax PAC bounds on the sample complexity of reinforcement
learning with a generative model. Machine Learning, 91(3), 325–349.
Azar, M. G., Osband, I., & Munos, R. (2017). Minimax regret bounds for reinforcement learning. In International
conference on machine learning (pp. 263–272). PMLR.
Baldacci, B., & Manziuk, I. (2020). Adaptive trading strategies across liquidity pools. arXiv preprint
arXiv:2008.07807.
Baldacci, B., Manziuk, I., Mastrolia, T., & Rosenbaum, M. (2019). Market making and incentives design in the
presence of a dark pool: A deep reinforcement learning approach. arXiv preprint arXiv:1912.01129.
Bao, W., & Liu, X.-y. (2019). Multi-agent deep reinforcement learning for liquidation strategy analysis. arXiv preprint
arXiv:1906.11046.
Basak, S., & Chabakauri, G. (2010). Dynamic mean-variance asset allocation. The Review of Financial Studies, 23(8),
2970–3016.
Basei, M., Guo, X., Hu, A., & Zhang, Y. (2021). Logarithmic regret for episodic continuous-time linear-quadratic
reinforcement learning over a finite-time horizon. Available at SSRN 3848428, 23(178), 1–34.
Beck, C. L., & Srikant, R. (2012). Error bounds for constant step-size Q-learning. Systems & Control Letters, 61(12),
1203–1208.
Bellemare, M. G., Naddaf, Y., Veness, J., & Bowling, M. (2013). The Arcade Learning Environment: An evaluation
platform for general agents. Journal of Artificial Intelligence Research, 47, 253–279.
Berry, D. A., & Fristedt, B. (1985). Bandit problems: Sequential allocation of experiments (Monographs on statistics
and applied probability) (Vol. 5, pp. 7). Chapman and Hall.
Bhandari, J., & Russo, D. (2019). Global optimality guarantees for policy gradient methods. arXiv preprint
arXiv:1906.01786.
Bhandari, J., Russo, D., & Singal, R. (2018). A finite time analysis of temporal difference learning with linear
function approximation. In Conference on learning theory (pp. 1691–1692). PMLR.
Bhatnagar, S. (2010). An actor–critic algorithm with function approximation for discounted cost constrained
Markov decision processes. Systems & Control Letters, 59(12), 760–766.
Bjork, T., & Murgoci, A. (2010). A general theory of Markovian time inconsistent stochastic control problems.
Available at SSRN 1694759.
Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3),
637–654.
Bradtke, S. J., & Barto, A. G. (1996). Linear least-squares algorithms for temporal difference learning. Machine
Learning, 22(1), 33–57.

Brafman, R. I., & Tennenholtz, M. (2002). R-max-a general polynomial time algorithm for near-optimal
reinforcement learning. Journal of Machine Learning Research, 3, 213–231.
Broadie, M., & Detemple, J. B. (2004). Anniversary article: Option pricing: Valuation models and applications.
Management Science, 50(9), 1145–1177.
Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. Quantitative Finance, 19(8), 1271–1291.
Cai, Q., Yang, Z., Lee, J., & Wang, Z. (2019). Neural temporal-difference learning converges to global optima. In
Advances in neural information processing systems.
Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997). The econometrics of financial markets. Princeton University
Press.
Cannelli, L., Nuti, G., Sala, M., & Szehr, O. (2020). Hedging using reinforcement learning: Contextual 𝑘-armed
bandit versus Q-learning. arXiv preprint arXiv:2007.01623.
Cao, J., Chen, J., Hull, J., & Poulos, Z. (2021). Deep hedging of derivatives using reinforcement learning. The Journal
of Financial Data Science, 3(1), 10–27.
Capponi, A., Olafsson, S., & Zariphopoulou, T. (2021). Personalized robo-advising: Enhancing investment through
client interaction. Management Science, 68(4), 2485–2512.
Carbonneau, A., & Godin, F. (2021). Equal risk pricing of derivatives with deep hedging. Quantitative Finance, 21(4),
593–608.
Cartea, Á., Jaimungal, S., & Penalva, J. (2015). Algorithmic and high-frequency trading. Cambridge University Press.
Cartea, Á., Jaimungal, S., & Sánchez-Betancourt, L. (2021). Deep reinforcement learning for algorithmic trading.
Available at SSRN.
Cayci, S., Satpathi, S., He, N., & Srikant, R. (2021). Sample complexity and overparameterization bounds for
projection-free neural TD learning. arXiv preprint arXiv:2103.01391.
Cen, S., Cheng, C., Chen, Y., Wei, Y., & Chi, Y. (2020). Fast global convergence of natural policy gradient methods
with entropy regularization. arXiv preprint arXiv:2007.06558.
Chakraborti, A., Toke, I. M., Patriarca, M., & Abergel, F. (2011). Econophysics review: I. empirical facts. Quantitative
Finance, 11(7), 991–1012.
Chan, N. T., & Shelton, C. (2001). An electronic market-maker (Technical Report). MIT.
Charpentier, A., Elie, R., & Remlinger, C. (2021). Reinforcement learning in economics and finance. Computational
Economics, 1–38.
Chen, J., & Jiang, N. (2019). Information-theoretic considerations in batch reinforcement learning. In International
conference on machine learning (pp. 1042–1051). PMLR.
Cheung, W. C., Simchi-Levi, D., & Zhu, R. (2019). Learning to optimize under non-stationarity. In Proceedings of
the 22nd international conference on artificial intelligence and statistics (pp. 1079–1087). PMLR.
Cheung, W. C., Simchi-Levi, D., & Zhu, R. (2020). Reinforcement learning for non-stationary Markov decision
processes: The blessing of (more) optimism. In International conference on machine learning (pp. 1843–1854).
PMLR.
Chow, Y., Ghavamzadeh, M., Janson, L., & Pavone, M. (2017). Risk-constrained reinforcement learning with
percentile risk criteria. The Journal of Machine Learning Research, 18(1), 6070–6120.
Chow, Y., Tamar, A., Mannor, S., & Pavone, M. (2015). Risk-sensitive and robust decision-making: A CVaR
optimization approach. In NIPS’15 (pp. 1522–1530). MIT Press.
Coache, A., & Jaimungal, S. (2021). Reinforcement learning with dynamic convex risk measures. arXiv preprint
arXiv:2112.13414.
Cong, L. W., Tang, K., Wang, J., & Zhang, Y. (2021). Alphaportfolio: Direct construction through deep reinforcement
learning and interpretable ai. SSRN Electronic Journal. https://doi.org/10.2139/ssrn
Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. Quantitative Finance, 1(2),
223–236.
Cont, R., & Kukanov, A. (2017). Optimal order placement in limit order markets. Quantitative Finance, 17(1), 21–39.
Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified approach. Journal of Financial
Economics, 7(3), 229–263.
Dabérius, K., Granat, E., & Karlsson, P. (2019). Deep execution-value and policy based reinforcement learning for
trading and beating market benchmarks. Available at SSRN 3374766.
Dabney, W., Ostrovski, G., & Barreto, A. (2020). Temporally-extended 𝜖-greedy exploration. arXiv preprint
arXiv:2006.01782.

Dai, B., Shaw, A., Li, L., Xiao, L., He, N., Liu, Z., Chen, J., & Song, L. (2018). SBEED: Convergent reinforcement
learning with nonlinear function approximation. In International conference on machine learning (pp. 1125–1134).
PMLR.
Dalal, G., Szörényi, B., Thoppe, G., & Mannor, S. (2018). Finite sample analyses for TD(0) with function
approximation. In 32th AAAI conference on artificial intelligence.
Dann, C., & Brunskill, E. (2015). Sample complexity of episodic fixed-horizon reinforcement learning. In NIPS’15
(pp. 2818–2826). MIT Press.
Dann, C., Lattimore, T., & Brunskill, E. (2017). Unifying PAC and regret: Uniform PAC bounds for episodic rein-
forcement learning. In Proceedings of the 31st international conference on neural information processing systems,
NIPS’17 (pp. 5717–5727).
Dann, C., Mansour, Y., Mohri, M., Sekhari, A., & Sridharan, K. (2022). Guarantees for epsilon-Greedy reinforce-
ment learning with function approximation. In International conference on machine learning (pp. 4666–4689).
PMLR.
Deng, Y., Bao, F., Kong, Y., Ren, Z., & Dai, Q. (2017). Deep direct reinforcement learning for financial signal
representation and trading. IEEE Transactions on Neural Networks and Learning Systems, 28(3), 653–664.
Ding, D., Wei, X., Yang, Z., Wang, Z., & Jovanovic, M. (2021). Provably efficient safe exploration via primal-dual
policy optimization. In International conference on artificial intelligence and statistics (pp. 3304–3312). PMLR.
Dixon, M., & Halperin, I. (2020). G-learner and girl: Goal based wealth management with reinforcement learning.
arXiv preprint arXiv:2002.10990.
Dixon, M. F., Halperin, I., & Bilokon, P. (2020). Machine learning in finance. Springer.
Du, J., Jin, M., Kolm, P. N., Ritter, G., Wang, Y., & Zhang, B. (2020). Deep reinforcement learning for option
replication and hedging. The Journal of Financial Data Science, 2(4), 44–57.
Du, X., Zhai, J., & Lv, K. (2016). Algorithm trading using Q-learning and recurrent reinforcement learning. Positions,
1, 1.
Dubrov, B. (2015). Monte Carlo simulation with machine learning for pricing American options and convertible
bonds. Available at SSRN 2684523.
Eriksson, H., & Dimitrakakis, C. (2019). Epistemic risk-sensitive reinforcement learning. arXiv preprint
arXiv:1906.06273.
Even-Dar, E., Mansour, Y., & Bartlett, P. (2003). Learning rates for Q-learning. Journal of Machine Learning
Research, 5(1), 1–25.
Fan, J., Ma, C., & Zhong, Y. (2021). A selective overview of deep learning. Statistical Science, 36, 264–290.
Fan, J., Wang, Z., Xie, Y., & Yang, Z. (2020). A theoretical analysis of deep Q-learning. In Learning for dynamics and
control (pp. 486–489). PMLR.
Farahmand, A. M., Ghavamzadeh, M., Szepesvàri, C., & Mannor, S. (2008). Regularized policy iteration. In
Advances in neural information processing systems 21 - Proceedings of the 2008 conference (pp. 441–448).
Fazel, M., Ge, R., Kakade, S. M., & Mesbahi, M. (2018). Global convergence of policy gradient methods for the linear
quadratic regulator. In International conference on machine learning (pp. 1467–1476). PMLR.
Fei, Y., Yang, Z., Chen, Y., Wang, Z., & Xie, Q. (2020). Risk-sensitive reinforcement learning: Near-optimal risk-
sample tradeoff in regret. In NeurIPS.
Fermanian, J.-D., Guéant, O., & Rachez, A. (2015). Agents’ behavior on multi-dealer-to-client bond trading platforms.
CREST, Center for Research in Economics and Statistics.
Figlewski, S. (1989). Options arbitrage in imperfect markets. The Journal of Finance, 44(5), 1289–1311.
Fischer, T. G. (2018). Reinforcement learning in financial markets—a survey (Technical Report). FAU Discussion
Papers in Economics.
François-Lavet, V., Henderson, P., Islam, R., Bellemare, M. G., & Pineau, J. (2018). An introduction to deep
reinforcement learning. Foundations and Trends in Machine Learning, 11(3-4), 219–354.
François-Lavet, V., Rabusseau, G., Pineau, J., Ernst, D., & Fonteneau, R. (2019). On overfitting and asymptotic bias
in batch reinforcement learning with partial observability. Journal of Artificial Intelligence Research, 65, 1–30.
Fu, Z., Yang, Z., & Wang, Z. (2021). Single-timescale actor-critic provably finds globally optimal policy. In
International conference on learning representations.
Gajane, P., Ortner, R., & Auer, P. (2018). A sliding-window algorithm for Markov decision processes with arbitrarily
changing rewards and transitions. arXiv preprint arXiv:1805.10066.

Ganchev, K., Nevmyvaka, Y., Kearns, M., & Vaughan, J. W. (2010). Censored exploration and the dark pool problem.
Communications of the ACM, 53(5), 99–107.
Ganesh, S., Vadori, N., Xu, M., Zheng, H., Reddy, P., & Veloso, M. (2019). Reinforcement learning for market making
in a multi-agent dealer market. arXiv preprint arXiv:1911.05892.
Gao, X., Xu, Z. Q., & Zhou, X. Y. (2020). State-dependent temperature control for langevin diffusions. arXiv preprint
arXiv:2011.07456.
Gao, Z., Han, Y., Ren, Z., & Zhou, Z. (2019). Batched multi-armed bandits problem. In Advances in Neural
Information Processing Systems (Vol. 32).
Garcelon, E., Ghavamzadeh, M., Lazaric, A., & Pirotta, M. (2020). Conservative exploration in reinforcement
learning. In International conference on artificial intelligence and statistics (pp. 1431–1441). PMLR.
Gašperov, B., & Kostanjčar, Z. (2021). Market making with signals through deep reinforcement learning. IEEE
Access, 9, 61611–61622.
Geist, M., Scherrer, B., & Pietquin, O. (2019). A theory of regularized Markov decision processes. In International
conference on machine learning (pp. 2160–2169). PMLR.
Giurca, A., & Borovkova, S. (2021). Delta hedging of derivatives using deep reinforcement learning. Available at
SSRN 3847272.
Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.
Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014).
Generative adversarial nets. Advances in Neural Information Processing Systems (Vol. 27).
Goodfellow, I., Warde-Farley, D., Mirza, M., Courville, A., & Bengio, Y. (2013). Maxout networks. In International
conference on machine learning (pp. 1319–1327). PMLR.
Gopalan, A., & Mannor, S. (2015). Thompson sampling for learning parameterized Markov decision processes. In
Conference on learning theory (pp. 861–898). PMLR.
Gordon, G. J. (1996). Stable fitted reinforcement learning. In Advances in neural information processing systems (pp.
1052–1058).
Grau-Moya, J., Leibfried, F., & Vrancx, P. (2018). Soft Q-learning with mutual-information regularization. In
International conference on learning representations.
Grinold, R. C., & Kahn, R. N. (2000). Active portfolio management. McGraw-Hill.
Gu, S., Lillicrap, T., Sutskever, I., & Levine, S. (2016). Continuous deep Q-learning with model-based acceleration.
In International conference on machine learning (pp. 2829–2838). PMLR.
Guéant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2012). Optimal portfolio liquidation with limit orders. SIAM
Journal on Financial Mathematics, 3(1), 740–764.
Guéant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2013). Dealing with the inventory risk: A solution to the market
making problem. Mathematics and Financial Economics, 7(4), 477–507.
Guéant, O., & Manziuk, I. (2019). Deep reinforcement learning for market making in corporate bonds: beating the
curse of dimensionality. Applied Mathematical Finance, 26(5), 387–452.
Guilbaud, F., & Pham, H. (2013). Optimal high-frequency trading with limit and market orders. Quantitative
Finance, 13(1), 79–94.
Guo, X., Hu, A., & Zhang, Y. (2021). Reinforcement learning for linear-convex models with jumps via stability
analysis of feedback controls. arXiv preprint arXiv:2104.09311.
Haarnoja, T., Tang, H., Abbeel, P., & Levine, S. (2017). Reinforcement learning with deep energy-based policies. In
International conference on machine learning (pp. 1352–1361). PMLR.
Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep rein-
forcement learning with a stochastic actor. In International conference on machine learning (pp. 1861–1870).
PMLR.
Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., &
Levine, S. (2018). Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905.
Hakansson, N. H. (1971). Multi-period mean-variance analysis: Toward a general theory of portfolio choice. The
Journal of Finance, 26(4), 857–884.
Halperin, I. (2019). The QLBS Q-learner goes NuQlear: Fitted Q iteration, inverse RL, and option portfolios.
Quantitative Finance, 19(9), 1543–1553.
Halperin, I. (2020). QLBS: Q-learner in the Black-Scholes (-Merton) worlds. The Journal of Derivatives, 28(1), 99–122.

Hambly, B., Xu, R., & Yang, H. (2021). Policy gradient methods for the noisy linear quadratic regulator over a finite
horizon. SIAM Journal on Control and Optimization, 59(5), 3359–3391.
Hasselt, H. (2010). Double Q-learning. In Advances in Neural Information Processing Systems (Vol. 23, pp. 2613–2621.
Hendricks, D., & Wilcox, D. (2014). A reinforcement learning extension to the Almgren-Chriss framework for opti-
mal trade execution. In 2014 IEEE Conference on computational intelligence for financial engineering & economics
(CIFEr) (pp. 457–464). IEEE.
Henrotte, P. (1993). Transaction costs and duplication strategies. Graduate School of Business, Stanford University.
Heston, S. (1993). A closed-form solution for options with stochastic volatility with applications to bond and
currency options. Review of Financial Studies, 6, 327–343.
Huang, N. E., Wu, M.-L., Qu, W., Long, S. R., & Shen, S. S. (2003). Applications of Hilbert–Huang transform to
non-stationary financial time series analysis. Applied Stochastic Models in Business and Industry, 19(3), 245–268.
Ian, O., Benjamin, V. R., & Daniel, R. (2013). (More) Efficient reinforcement learning via posterior sampling. In
Proceedings of the 26th international conference on neural information processing systems, NIPS’13, (Vol. 2, pp.
3003–3011).
Jaimungal, S., Pesenti, S. M., Wang, Y. S., & Tatsat, H. (2021). Robust risk-aware reinforcement learning. Available
at SSRN 3910498, 13(1), 213–226.
Jeong, G., & Kim, H. Y. (2019). Improving financial trading decisions using deep Q-learning: Predicting the number
of shares, action strategies, and transfer learning. Expert Systems with Applications, 117, 125–138.
Jia, Y., & Zhou, X. Y. (2021). Policy gradient and actor-critic learning in continuous time and space: Theory and
algorithms. arXiv preprint arXiv:2111.11232.
Jia, Y., & Zhou, X. Y. (2022). Policy evaluation and temporal-difference learning in continuous time and space: A
martingale approach. Journal of Machine Learning Research, 23(154), 1–55.
Jiang, J., Kelly, B. T., & Xiu, D. (2020). (Re-) Imag (in) ing price trends [Research paper]. Chicago Booth.
Jiang, Z., Xu, D., & Liang, J. (2017). A deep reinforcement learning framework for the financial portfolio
management problem. arXiv preprint arXiv:1706.10059.
Jin, C., Allen-Zhu, Z., Bubeck, S., & Jordan, M. I. (2018). Is Q-learning provably efficient? In Advances in neural
information processing systems (Vol. 31).
Jin, C., Yang, Z., Wang, Z., & Jordan, M. I. (2020). Provably efficient reinforcement learning with linear function
approximation. In Conference on learning theory (pp. 2137–2143). PMLR.
Kakade, S. M. (2001). A natural policy gradient. In Advances in neural information processing systems (Vol. 14).
Karpe, M., Fang, J., Ma, Z., & Wang, C. (2020). Multi-agent reinforcement learning in a realistic limit order book
market simulation. In Proceedings of the first ACM international conference on AI in finance, ICAIF’20.
Ke, T. T., Shen, Z.-J. M., & Villas-Boas, J. M. (2016). Search for information on multiple products. Management
Science, 62(12), 3576–3603.
Kearns, M., & Singh, S. (2002). Near-optimal reinforcement learning in polynomial time. Machine Learning, 49(2),
209–232.
Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In Proceedings of the 3rd international
conference on learning representations (ICLR).
Klöppel, S., & Schweizer, M. (2007). Dynamic indifference valuation via convex risk measures. Mathematical
Finance, 17(4), 599–627.
Koenig, S., & Simmons, R. G. (1993). Complexity analysis of real-time reinforcement learning. In AAAI (pp. 99–107).
Kolm, P. N., & Ritter, G. (2019). Dynamic replication and hedging: A reinforcement learning approach. The Journal
of Financial Data Science, 1(1), 159–171.
Kolm, P. N., & Ritter, G. (2020). Modern perspectives on reinforcement learning in finance. The Journal of Machine
Learning in Finance, 1, 28.
Konda, V. (2002). Actor-critic algorithms (PhD thesis). MIT.
Konda, V. R., & Tsitsiklis, J. N. (2000). Actor-critic algorithms. In Advances in neural information processing systems
(pp. 1008–1014).
Kühn, C., & Stroh, M. (2010). Optimal portfolios of a small investor in a limit order market: A shadow price
approach. Mathematics and Financial Economics, 3(2), 45–72.
Kumar, H., Koppel, A., & Ribeiro, A. (2019). On the sample complexity of actor-critic method for reinforcement
learning with function approximation. arXiv preprint arXiv:1910.08412.

Lagoudakis, M. G., & Parr, R. (2003). Least-squares policy iteration. The Journal of Machine Learning Research, 4,
1107–1149.
Lakshminarayanan, C., & Szepesvari, C. (2018). Linear stochastic approximation: How far does constant step-size
and iterate averaging go? In International conference on artificial intelligence and statistics (pp. 1347–1355). PMLR.
Lattimore, T., & Hutter, M. (2012). PAC bounds for discounted MDPs. In International conference on algorithmic
learning theory (pp. 320–334). Springer.
Lattimore, T., Szepesvari, C., & Weisz, G. (2020). Learning with good feature representations in bandits and in RL
with a generative model. In International conference on machine learning (pp. 5662–5670). PMLR.
Leal, L., Laurière, M., & Lehalle, C.-A. (2020). Learning a functional control for high-frequency finance. arXiv
preprint arXiv:2006.09611.
LeCun, Y., & Bengio, Y. (1995). Convolutional networks for images, speech, and time series. In The handbook of
brain theory and neural networks (Vol. 3361). MIT Press.
Lehalle, C.-A., & Laruelle, S. (2018). Market microstructure in practice. World Scientific.
Leland, H. E. (1985). Option pricing and replication with transactions costs. The Journal of Finance, 40(5), 1283–1301.
Li, D., & Ng, W.-L. (2000). Optimal dynamic portfolio selection: Multiperiod mean-variance formulation.
Mathematical Finance, 10(3), 387–406.
Li, L. (2009). A unifying framework for computational reinforcement learning theory. Rutgers—The State University
of New Jersey—New Brunswick.
Li, Y., Szepesvari, C., & Schuurmans, D. (2009). Learning exercise policies for American options. In Artificial
intelligence and statistics (pp. 352–359). PMLR.
Liang, Z., Chen, H., Zhu, J., Jiang, K., & Li, Y. (2018). Adversarial deep reinforcement learning in portfolio
management. arXiv preprint arXiv:1808.09940.
Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous
control with deep reinforcement learning. In 4th international conference on learning representations (ICLR).
Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine
Learning, 8(3-4), 293–321.
Lin, S., & Beling, P. A. (2020). An end-to-end optimal trade execution framework based on proximal policy
optimization. In IJCAI (pp. 4548–4554).
Liu, B., Cai, Q., Yang, Z., & Wang, Z. (2019). Neural trust region/proximal policy optimization attains globally
optimal policy. In Advances in neural information processing systems (Vol. 32).
Liu, X.-Y., Xia, Z., Rui, J., Gao, J., Yang, H., Zhu, M., Wang, C. D., Wang, Z., & Guo, J. (2022). FinRL-Meta: Market
environments and benchmarks for data-driven financial reinforcement learning. arXiv preprint arXiv:2211.03107.
Liu, X.-Y., Yang, H., Gao, J., & Wang, C. D. (2021). FinRL: Deep reinforcement learning framework to automate
trading in quantitative finance. In Proceedings of the second ACM international conference on AI in finance (pp.
1–9).
Liu, Y., Zhang, K., Basar, T., & Yin, W. (2020). An improved analysis of (variance-reduced) policy gradient and
natural policy gradient methods. In NeurIPS.
Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach.
The Review of Financial Studies, 14(1), 113–147.
Mao, W., Zhang, K., Zhu, R., Simchi-Levi, D., & Başar, T. (2020). Model-free non-stationary RL: Near-optimal regret
and applications in multi-agent RL and inventory control. arXiv preprint arXiv:2010.03161.
Markowitz, H. M. (1952). Portfolio selection. Journal of Finance, 7(1), 77–91.
massoud Farahmand, A., Ghavamzadeh, M., Szepesvàri, C., & Mannor, S. (2009). Regularized fitted Q-iteration for
planning in continuous-space Markovian decision problems. In 2009 American control conference (pp. 725–730).
IEEE.
Mei, J., Xiao, C., Szepesvari, C., & Schuurmans, D. (2020). On the global convergence rates of softmax policy
gradient methods. In International conference on machine learning (pp. 6820–6829). PMLR.
Melo, F. S., & Ribeiro, M. I. (2007). Q-learning with linear function approximation. In International conference on
computational learning theory (pp. 308–322). Springer.
Meng, T. L., & Khushi, M. (2019). Reinforcement learning in financial markets. Data, 4(3), 110.
Merton, R. C. (1973). Theory of rational option pricing. The Bell Journal of Economics and Management Science, 4,
141–183.

Merton, R. C., & Samuelson, P. A. (1974). Fallacy of the log-normal approximation to optimal portfolio decision-
making over many periods. Journal of Financial Economics, 1(1), 67–94.
Mihatsch, O., & Neuneier, R. (2002). Risk-sensitive reinforcement learning. Machine Learning, 49(2), 267–290.
Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016).
Asynchronous methods for deep reinforcement learning. In International conference on machine learning (pp.
1928–1937). PMLR.
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing
Atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland,
A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg,
S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529–533.
Moody, J., Wu, L., Liao, Y., & Saffell, M. (1998). Performance functions and reinforcement learning for trading
systems and portfolios. Journal of Forecasting, 17(5-6), 441–470.
Mosavi, A., Faghan, Y., Ghamisi, P., Duan, P., Ardabili, S. F., Salwana, E., & Band, S. S. (2020). Comprehensive
review of deep reinforcement learning methods and applications in economics. Mathematics, 8(10), 1640.
Mossin, J. (1968). Optimal multiperiod portfolio policies. The Journal of Business, 41(2), 215–229.
Nesterov, Y. E. (1983). A method for solving the convex programming problem with convergence rate o (1/k 2). Dokl.
Akad. Nauk SSSR, 269, 543–547.
Nevmyvaka, Y., Feng, Y., & Kearns, M. (2006). Reinforcement learning for optimized trade execution. In Proceedings
of the 23rd international conference on machine learning (pp. 673–680).
Ning, B., Ling, F. H. T., & Jaimungal, S. (2018). Double deep Q-learning for optimal execution. arXiv preprint
arXiv:1812.06600.
Obizhaeva, A. A., & Wang, J. (2013). Optimal trading strategy and supply/demand dynamics. Journal of Financial
Markets, 16(1), 1–32.
Ouyang, Y., Gagrani, M., Nayyar, A., & Jain, R. (2017). Learning unknown Markov decision processes: A Thompson
sampling approach. In Advances in neural information processing systems (Vol. 30).
Pan, S. J., & Yang, Q. (2009). A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering,
22(10), 1345–1359.
Papini, M., Binaghi, D., Canonaco, G., Pirotta, M., & Restelli, M. (2018). Stochastic variance-reduced policy gradient.
In International conference on machine learning (pp. 4026–4035). PMLR.
Park, H., Sim, M. K., & Choi, D. G. (2020). An intelligent financial portfolio trading strategy using deep Q-learning.
Expert Systems with Applications, 158, 113573.
Patel, Y. (2018). Optimizing market making using multi-agent reinforcement learning. arXiv preprint
arXiv:1812.10252.
Pedersen, J. L., & Peskir, G. (2017). Optimal mean-variance portfolio selection. Mathematics and Financial
Economics, 11(2), 137–160.
Pendharkar, P. C., & Cusatis, P. (2018). Trading financial indices with reinforcement learning agents. Expert Systems
with Applications, 103, 1–13.
Perold, A. F. (1988). The implementation shortfall: Paper versus reality. Journal of Portfolio Management, 14(3), 4–9.
Peters, J., & Schaal, S. (2008). Natural actor-critic. Neurocomputing, 71(7-9), 1180–1190.
Pomatto, L., Strack, P., & Tamuz, O. (2018). The cost of information. arXiv preprint arXiv:1812.04211.
Powell, W. B. (2021). Reinforcement learning and stochastic optimization. John Wiley & Sons.
Preis, T. (2011). Price-time priority and pro rata matching in an order book model of financial markets. In
Econophysics of order-driven markets (pp. 65–72). Springer.
Puterman, M. L. (2014). Markov decision processes: Discrete stochastic dynamic programming. John Wiley & Sons.
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised
multitask learners. OpenAI Blog, 1(8), 9.
Ren, Z., & Zhou, Z. (2020). Dynamic batch learning in high-dimensional sparse linear contextual bandits. arXiv
preprint arXiv:2008.11918.
Riedmiller, M. (2005). Neural fitted Q iteration—first experiences with a data efficient neural reinforcement
learning method. In European conference on machine learning (pp. 317–328). Springer.
Robbins, H. (1952). Some aspects of the sequential design of experiments. Bulletin of the American Mathematical
Society, 58(5), 527–535.

Samuelson, P. A. (1975). Lifetime portfolio selection by dynamic stochastic programming. In Stochastic optimization
models in finance (pp. 517–524). Academic Press.
Sato, Y. (2019). Model-free reinforcement learning for financial portfolios: A brief survey. arXiv preprint
arXiv:1904.04973.
Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. In
International conference on machine learning (pp. 1889–1897). PMLR.
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms.
arXiv preprint arXiv:1707.06347.
Sewak, M. (2019). Policy-based reinforcement learning approaches. In Deep reinforcement learning (pp. 127–140).
Springer.
Shani, L., Efroni, Y., & Mannor, S. (2020). Adaptive trust region policy optimization: Global convergence and faster
rates for regularized MDPs. In Proceedings of the AAAI conference on artificial intelligence (Vol. 34, pp. 5668–5675).
Sharpe, W. F. (1966). Mutual fund performance. The Journal of Business, 39(1), 119–138.
Shen, Y., Huang, R., Yan, C., & Obermayer, K. (2014). Risk-averse reinforcement learning for algorithmic trading. In
2014 IEEE conference on computational intelligence for financial engineering & economics (CIFEr) (pp. 391–398).
IEEE.
Shen, Y., Tobia, M. J., Sommer, T., & Obermayer, K. (2014). Risk-sensitive reinforcement learning. Neural
Computation, 26(7), 1298–1328.
Shen, Z., Ribeiro, A., Hassani, H., Qian, H., & Mi, C. (2019). Hessian aided policy gradient. In International
conference on machine learning (pp. 5729–5738). PMLR.
Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel,
T., Lillicrap, T., Simonyan, K., & Hassabis, D. (2017). Mastering chess and shogi by self-play with a general
reinforcement learning algorithm. arXiv preprint arXiv:1712.01815.
Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014). Deterministic policy gradient
algorithms. In International conference on machine learning (pp. 387–395). PMLR.
Simchi-Levi, D., & Xu, Y. (2020). Bypassing the monster: A faster and simpler optimal algorithm for contextual
bandits under realizability. Available at SSRN 3562765, 47(3), 1904–1931.
Sortino, F. A., & Price, L. N. (1994). Performance measurement in a downside risk framework. The Journal of
Investing, 3(3), 59–64.
Spooner, T., Fearnley, J., Savani, R., & Koukorinis, A. (2018). Market making via reinforcement learning. In
International foundation for autonomous agents and multiagent systems, AAMAS’18 (pp. 434–442).
Spooner, T., & Savani, R. (2020). Robust Market Making via Adversarial Reinforcement Learning. In Proceedings
of the 29th international joint conference on artificial intelligence, IJCAI-20 (pp. 4590–4596).
Steinbach, M. (2001). Markowitz revisited: Mean-variance models in financial portfolio analysis. SIAM Review, 43,
31–85.
Strehl, A. L., Li, L., & Littman, M. L. (2009). Reinforcement learning in finite MDPs: PAC analysis. Journal of
Machine Learning Research, 10(11), 2413–2444.
Strehl, A. L., & Littman, M. L. (2005). A theoretical analysis of model-based interval estimation. In Proceedings
of the 22nd international conference on machine learning ICML’05 (pp. 856–863). Association for Computing
Machinery.
Strotz, R. H. (1955). Myopia and inconsistency in dynamic utility maximization. The Review of Economic Studies,
23(3), 165–180.
Sutskever, I., Martens, J., & Hinton, G. E. (2011). Generating text with recurrent neural networks. In ICML.
Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT Press.
Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000). Policy gradient methods for reinforcement
learning with function approximation. In Advances in neural information processing systems (pp. 1057–1063).
Szita, I., & Szepesvári, C. (2010). Model-based reinforcement learning with nearly tight exploration complexity
bounds. In ICML (pp. 1031–1038).
Tamar, A., Chow, Y., Ghavamzadeh, M., & Mannor, S. (2015). Policy gradient for coherent risk measures. In
Advances in neural information processing systems (Vol. 28).
Tang, W., Zhang, P. Y., & Zhou, X. Y. (2021). Exploratory HJB equations and their convergence. arXiv preprint
arXiv:2109.10269.

Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence
of two samples. Biometrika, 25(3/4), 285–294.
Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent
magnitude. COURSERA: Neural Networks for Machine Learning, 4(2), 26–31.
Torrey, L., & Shavlik, J. (2010). Transfer learning. In Handbook of research on machine learning applications and
trends: Algorithms, methods, and techniques (pp. 242–264). IGI Global.
Touati, A., & Vincent, P. (2020). Efficient learning in non-stationary linear Markov decision processes. arXiv
preprint arXiv:2010.12870.
Vadori, N., Ganesh, S., Reddy, P., & Veloso, M. (2020). Risk-sensitive reinforcement learning: A martingale
approach to reward uncertainty. arXiv preprint arXiv:2006.12686.
Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. In Proceedings
of the AAAI Conference on Artificial Intelligence (Vol. 30).
Vigna, E. (2016). On time consistency for mean-variance portfolio selection. Collegio Carlo Alberto Notebook, 476.
Vinyals, O., Babuschkin, I., Czarnecki, W. M., Mathieu, M., Dudzik, A., Chung, J., Choi, D. H., Powell, R., Ewalds, T.,
Georgiev, P., Oh, J., Horgan, D., Kroiss, M., Danihelka, I., Huang, A., Sifre, L., Cai, T., Agapiou, J. P., Jaderberg,
M., ... Silver, D. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature,
575(7782), 350–354.
Von Luxburg, U., & Schölkopf, B. (2011). Statistical learning theory: Models, concepts, and results. In Handbook of
the history of logic, (Vol. 10, pp. 651–706). Elsevier.
Wang, H. (2019). Large scale continuous-time mean-variance portfolio allocation via reinforcement learning.
Available at SSRN 3428125.
Wang, H., & Yu, S. (2021). Robo-advising: Enhancing investment with inverse optimization and deep reinforcement
learning. arXiv preprint arXiv:2105.09264.
Wang, H., Zariphopoulou, T., & Zhou, X. (2020). Exploration versus exploitation in reinforcement learning: A
stochastic control approach. Journal of Machine Learning Research, 21, 1–34.
Wang, H., & Zhou, X. Y. (2020). Continuous-time mean–variance portfolio selection: A reinforcement learning
framework. Mathematical Finance, 30(4), 1273–1308.
Wang, L., Cai, Q., Yang, Z., & Wang, Z. (2020). Neural policy gradient methods: Global optimality and rates of
convergence. In International conference on learning representations.
Wang, Y., Dong, K., Chen, X., & Wang, L. (2020). Q-learning with UCB exploration is sample efficient for infinite-
horizon MDP. In International conference on learning representations.
Wang, Z., Bapst, V., Heess, N., Mnih, V., Munos, R., Kavukcuoglu, K., & Freitas, N. (2017). Sample efficient actor-
critic with experience replay. In International conference on learning representations (ICLR) (pp. 1–13).
Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine Learning, 8(3-4), 279–292.
Wei, C.-Y., Jahromi, M. J., Luo, H., Sharma, H., & Jain, R. (2020). Model-free reinforcement learning in infinite-
horizon average-reward Markov decision processes. In International conference on machine learning (pp. 10170–
10180). PMLR.
Wei, C.-Y., & Luo, H. (2021). Non-stationary reinforcement learning without prior knowledge: An optimal black-box
approach. In Conference on learning theory (pp. 4300–4354). PMLR.
Wei, H., Wang, Y., Mangu, L., & Decker, K. (2019). Model-based reinforcement learning for predictions and control
for limit order books. arXiv preprint arXiv:1910.03743.
Wiese, M., Knobloch, R., Korn, R., & Kretschmer, P. (2020). Quant GANs: Deep generation of financial time series.
Quantitative Finance, 20(9), 1419–1440.
Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.
Machine Learning, 8(3), 229–256.
Wu, Y., Shariff, R., Lattimore, T., & Szepesvári, C. (2016). Conservative bandits. In International conference on
machine learning (pp. 1254–1262). PMLR.
Xiao, H., Zhou, Z., Ren, T., Bai, Y., & Liu, W. (2020). Time-consistent strategies for multi-period mean-variance
portfolio optimization with the serially correlated returns. Communications in Statistics-Theory and Methods,
49(12), 2831–2868.
Xiong, H., Xu, T., Liang, Y., & Zhang, W. (2021). Non-asymptotic convergence of Adam-type reinforcement learn-
ing algorithms under Markovian sampling. Proceedings of the AAAI Conference on Artificial Intelligence, 35(12),
10460–10468.

Xiong, H., Zhao, L., Liang, Y., & Zhang, W. (2020). Finite-time analysis for double Q-learning. In Advances in neural
information processing systems (Vol. 33).
Xiong, Z., Liu, X.-Y., Zhong, S., Yang, H., & Walid, A. (2018). Practical deep reinforcement learning approach for
stock trading. arXiv preprint arXiv:1811.07522.
Xu, P., Gao, F., & Gu, Q. (2020a). An improved convergence analysis of stochastic variance-reduced policy gradient.
In Uncertainty in artificial intelligence (pp. 541–551). PMLR.
Xu, P., Gao, F., & Gu, Q. (2020b). Sample efficient policy gradient methods with recursive variance reduction. In
International conference on learning representations.
Xu, P., & Gu, Q. (2020). A finite-time analysis of Q-learning with neural network function approximation. In
International conference on machine learning (pp. 10555–10565). PMLR.
Xu, T., Wang, Z., & Liang, Y. (2020a). Improving sample complexity bounds for (natural) actor-critic algorithms.
In Advances in neural information processing systems (Vol. 33, pp. 4358–4369).
Xu, T., Wang, Z., & Liang, Y. (2020b). Non-asymptotic convergence analysis of two time-scale (natural) actor-critic
algorithms. arXiv preprint arXiv:2005.03557.
Xu, T., Yang, Z., Wang, Z., & Liang, Y. (2021). Doubly robust off-policy actor-critic: Convergence and optimality.
arXiv preprint arXiv:2102.11866.
Yang, H., Liu, X.-Y., & Wu, Q. (2018). A practical machine learning approach for dynamic stock recommendation.
In 2018 17th IEEE international conference on trust, security and privacy in computing and communications/12th
IEEE international conference on big data science and engineering (TrustCom/BigDataSE) (pp. 1693–1697). IEEE.
Yang, L., & Wang, M. (2019). Sample-optimal parametric Q-learning using linearly additive features. In
International conference on machine learning (pp. 6995–7004). PMLR.
Yang, L., & Wang, M. (2020). Reinforcement learning in feature space: Matrix bandit, kernels, and regret bound.
In International conference on machine learning (pp. 10746–10756). PMLR.
Yang, R., Sun, X., & Narasimhan, K. (2019). A generalized algorithm for multi-objective reinforcement learning
and policy adaptation. In Advances in neural information processing systems (Vol. 32).
Ye, Z., Deng, W., Zhou, S., Xu, Y., & Guan, J. (2020). Optimal trade execution based on deep deterministic policy
gradient. In Database systems for advanced applications (pp. 638–654). Springer International Publishing.
Yu, M., & Sun, S. (2020). Policy-based reinforcement learning for time series anomaly detection. Engineering
Applications of Artificial Intelligence, 95, 103919.
Yu, P., Lee, J. S., Kulyatin, I., Shi, Z., & Dasgupta, S. (2019). Model-based deep reinforcement learning for dynamic
portfolio optimization. arXiv preprint arXiv:1901.08740.
Yu, S., Wang, H., & Dong, C. (2020). Learning risk preferences from investment portfolios using inverse
optimization. arXiv preprint arXiv:2010.01687.
Zhang, G., & Chen, Y. (2020). Reinforcement learning for optimal market making with the presence of rebate.
Available at SSRN 3646753.
Zhang, J., Kim, J., O’Donoghue, B., & Boyd, S. (2021). Sample efficient reinforcement learning with REINFORCE.
In Proceedings of the AAAI conference on artificial intelligence (Vol. 35, pp. 10887–10895).
Zhang, K., Koppel, A., Zhu, H., & Basar, T. (2020). Global convergence of policy gradient methods to (almost) locally
optimal policies. SIAM Journal on Control and Optimization, 58(6), 3586–3612.
Zhang, Z., Zohren, S., & Roberts, S. (2020). Deep reinforcement learning for trading. The Journal of Financial Data
Science, 2(2), 25–40.
Zhao, M., & Linetsky, V. (2021). High frequency automated market making algorithms with adverse selection risk
control via reinforcement learning. In Proceedings of the second ACM international conference on AI in finance
(pp. 1–9).
Zheng, L., & Ratliff, L. (2020). Constrained upper confidence reinforcement learning. In Learning for dynamics and
control (pp. 620–629). PMLR.
Zhou, D., Chen, J., & Gu, Q. (2020). Provable multi-objective reinforcement learning with generative models. arXiv
preprint arXiv:2011.10134.
Zhou, X. Y., & Li, D. (2000). Continuous-time mean-variance portfolio selection: A stochastic LQ framework.
Applied Mathematics and Optimization, 42(1), 19–33.
Zivot, E. (2017). Introduction to computational finance and financial econometrics. Chapman & Hall CRC.
Zou, S., Xu, T., & Liang, Y. (2019). Finite-sample analysis for SARSA with linear function approximation. In
Advances in neural information processing systems (Vol. 32, pp. 8668–8678).

How to cite this article: Hambly, B., Xu, R., & Yang, H. (2023). Recent advances in
reinforcement learning in finance. Mathematical Finance, 33, 437–503.
https://doi.org/10.1111/mafi.12382
