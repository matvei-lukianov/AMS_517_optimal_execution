# Policy Gradient Methods for the Noisy Linear Quadratic Regulator over a Finite Horizon
Ben Hambly ∗
Renyuan Xu ∗
Huining Yang†∗
November 20, 2020
## Abstract
We explore reinforcement learning methods for ﬁnding the optimal policy in the linear quadratic
regulator (LQR) problem. In particular we consider the convergence of policy gradient methods in
the setting of known and unknown parameters. We are able to produce a global linear convergence
guarantee for this approach in the setting of ﬁnite time horizon and stochastic state dynamics under
weak assumptions. The convergence of a projected policy gradient method is also established in
order to handle problems with constraints. We illustrate the performance of the algorithm with two
examples. The ﬁrst example is the optimal liquidation of a holding in an asset. We show results for
the case where we assume a model for the underlying dynamics and where we apply the method to
the data directly. The empirical evidence suggests that the policy gradient method can learn the
global optimal solution for a larger class of stochastic systems containing the LQR framework and
that it is more robust with respect to model mis-speciﬁcation when compared to a model-based
approach. The second example is an LQR system in a higher dimensional setting with synthetic
data.
## 1 Introduction
The Linear Quadratic Regulator (LQR) problem is one of the most fundamental in optimal control
theory. Its aim is to ﬁnd a control for a linear dynamical system, that is the dynamics of the state of the
system is described by a linear function of the current state and input, subject to a quadratic cost. It is
an important problem for a number of reasons: (1) the LQR problem is one of the few optimal control
problems for which there exists a closed-form analytical representation of the optimal feedback control;
(2) when the dynamics are nonlinear and hard to analyze, a LQR approximation may be obtained as
a local expansion and provide an approximation that is provably close to the original problem; (3) the
LQR has been used in a wide variety of applications. In particular, in the set-up of ﬁxed time horizon
and stochastic dynamics, applications include portfolio optimization [3] and optimal liquidation [8] in
ﬁnance, resource allocation in energy markets [39, 43], and biological movement systems [34].
Until recently much of the work on the LQR problem has focused on solving for the optimal controls
under the assumption that the model parameters are fully known. See the book of Anderson and Moore
[10] for an introduction to the LQR problem with known parameters. However, assuming that the
controller has access to all the model parameters is not realistic for many applications, and this has lead
to the exploration of learning approaches to the problem. We consider reinforcement learning (RL),
one of the three basic machine learning paradigms (alongside supervised learning and unsupervised
learning). Unlike the situation with full information on the model parameters, RL is learning to make
∗Mathematical Institute, University of Oxford. Email: {hambly, xur, yang}@maths.ox.ac.uk
†Supported by the EPSRC Centre for Doctoral Training in Industrially Focused Mathematical Modelling
(EP/L015803/1) in collaboration with BP plc.

decisions via trial and error, through interactions with the (partially) unknown environment. In RL,
an agent takes an action and receives a reinforcement signal in terms of a numerical reward, which
encodes the outcome of her action. In order to maximize the accumulated reward over time, the agent
learns to select her actions based on her past experiences (exploitation) and/or by making new choices
(exploration). There are two popular approaches in RL to handle the LQR with unknown parameters:
the model-based approach and the model-free approach.
In the paradigm of the model-based approach, the controller estimates the unknown model parame-
ters and then constructs a control policy based on the estimated parameters. The classical approach is
the certainty equivalence principle [11]: the unknown parameters are estimated using observations (or
samples), and a control policy is then designed by treating the estimated parameters as the truth. In
the ﬁrst step, the unknown model parameters can be estimated by standard statistical methods such
as least-square minimization [21]. The second step is to show that when the estimated parameters
are accurate enough, the policy using the “plug-in” estimates enjoys good theoretical guarantees of
being close to optimal. See [21] and [25] for the optimal gap and sample complexities along this line
and see [23] for the sample complexity with distributed robust learning. Another line of work in the
model-based regime focuses on uncertainty quantiﬁcation. The controller updates their posterior belief
or the conﬁdence bounds on the unknown model parameters and then makes decisions in an online
manner, see [1, 2, 22, 31, 38].
Another recently developed approach is the model-free approach, where the controller learns the
optimal policy directly via interacting with the system, without inferring the model parameters. As
the optimal policy in the LQR problem is a linear function of the state, the aim is to determine this
linear function. This is equivalent to learning a set of parameters in matrix form, called the policy
matrix. One natural way to achieve this goal is to apply the gradient descent method in the parameter
space of the policy matrix, also referred to as the policy gradient method. In particular, the policy
gradient method computes the gradient of the cost function with respect to the policy matrix and then
updates the policy in the steepest decent direction to ﬁnd the optimal policy. The paper [24] was the
ﬁrst to show that policy gradients converge to the global optimal solution with polynomial (in the
relevant quantities) sample complexity. However, [24] focuses on the case where the only noise in the
system is in the initial state, and the rest of the state transitions are deterministic. There are other
methods that fall into the category of the model-free approach, including the Actor-Critic method [44]
and least-squares temporal diﬀerence learning [42].
Compared to the model-based approach, which strongly relies on the assumption that the stochastic
system lies within the LQR framework and may, in practice, suﬀer from model mis-speciﬁcation, the
execution of the model-free algorithm does not rely on the assumptions of the model. It has been
shown that the policy gradient method can learn the global optimal solution, not only for the LQR
framework, but also for a more general class of deterministic systems in the setting of an inﬁnite time
horizon [14]. Thus the advantage of the model-free approach is that it is more robust against model
mis-speciﬁcation compared to the model-based approach.
Our Contributions.
We now summarize our contributions. Motivated by many real-word decision-
making problems with a ﬁxed deadline and uncertainty in the underlying dynamics, such as the optimal
liquidation problem that we discuss in Section 2, we extend the framework of [24] by incorporating
a ﬁnite time horizon and sub-Gaussian noise (which includes Gaussian noise as a special case). In
particular, we provide a global linear convergence guarantee and a polynomial sample complexity
guarantee for the policy gradient method in this setting with both known parameters (Theorem 3.3)
and unknown parameters (Theorem 4.4).
The analysis with known parameters paves the way for
learning LQR with unknown parameters. In addition, numerically solving the Riccati equation with
known parameters in high dimensions may suﬀer from computational inaccuracy. The policy gradient
method provides a direct way of searching for the optimal solution with known parameters in this

case, which may be of separate interest. Note that the optimal policy is time-invariant for the LQR
with inﬁnite time horizon, whereas the optimal policy is time-dependent with ﬁnite time horizon and
hence harder to learn in general. With noise in the dynamics, we need more careful choices of the
hyper-parameters to retrieve compatible sample complexities with noisy observations. In addition,
when optimal polices need to satisfy certain constraints, we provide a global convergence result for the
projected policy gradient method in Theorem 4.5. This is required in the context of our application to
the optimal liquidation problem.
We will formulate the optimal liquidation problem over a ﬁxed horizon as a noisy LQR problem
which is essentially the classical Almgren-Chriss formulation [8]. The performance of the algorithm
on NASDAQ ITCH data is assessed. As well as using the method within this modelling approach, we
also consider the performance of the policy gradient method when applied directly to the data with an
appropriate cost function. This improves the performance of the LQR/Almgren-Chriss solution and
shows promising results for the use of the policy gradient method for problems that are ‘close’ to the
LQR framework.
1.1
Related Work
Policy Gradient Methods for LQR Problems.
Since the policy gradient method is the main
focus of our paper, here we provide a review of the previous theoretical work on this method in various
LQR settings and extensions. The ﬁrst global convergence result for the policy gradient method to
learn the optimal policy for LQR problems was developed in [24] in the setting of inﬁnite horizon and
deterministic dynamics. The work of [24] was extended in [14] to give global optimality guarantees of
policy gradient methods for a larger class of control problem that includes the linear-quadratic case.
In particular, this class of control problem satisﬁes a closure condition under policy improvement and
convexity of policy improvement steps. The paper [15] considers policy gradient methods for LQR
problems in terms of optimizing a real valued matrix function over the set of feedback gains. The
extension of the policy gradient method to continuous-time can be found in [16]. All of these methods
are in the inﬁnite horizon setting and without the addition of noise in the dynamics.
There has been some work on the case of noisy dynamics, but all in the setting of inﬁnite horizon. In
[27] the problem with a multiplicative noise was discussed, using a relatively straightforward extension
of the deterministic dynamics considered in the original framework.
In the case of additive noise
[32] studies the global convergence of policy gradient and other learning algorithms for the LQR over
an inﬁnite time-horizon and with Gaussian noise.
In particular, the policy considered in [32] is a
randomized policy with Gaussian distribution. There is also [35] which studies derivative-free (zeroth-
order) policy optimization methods for the LQR with bounded additive noise. Finally some other
contributions can be found in [17, 45] for zero-sum LQR games and [18, 29] for mean-ﬁeld LQR games.
There are several major technical diﬀerences compared to [24]. Due to the time-dependent nature
of the admissible policies over a ﬁnite-horizon and randomness from the system noise, we need more
careful handling of the system dynamics during the training to ensure they are well deﬁned. We also
need conditions to guarantee the gradient dominant condition, and for the proper sample size to get
a good estimate on the gradient of the cost function with high probability. See the more detailed
discussion in Remark 4.11.
Optimal Liquidation.
An early mathematical framework for the optimal liquidation problem is due
to Almgren and Chriss [8]. In this problem a trader is required to liquidate a portfolio of shares over
a ﬁxed horizon. The selling of a large number of shares at once has both temporary and permanent
impacts on the share price causing it to decrease.
The trader therefore wishes to ﬁnd a trading
strategy which maximizes their return from, or alternatively, minimizes the cost of, the liquidation of
the portfolio subject to a given level of risk.

This problem has been considered in many papers and extended in many directions. See for instance
[5], [7] and [26]. We will cast this as an LQR problem and show how the policy gradient method is a
powerful tool for solving this problem even without assumptions on the model.
More recently techniques from reinforcement learning have been applied to the optimal liquidation
problem.
The ﬁrst paper to do this was [36] where the authors showed promising results for this
approach by designing a Q-learning based algorithm to optimally select price levels and passively
place limit orders. This was further developed in [30] which designed a Q-learning based algorithm
for liquidation within the standard Almgren-Chriss framework. For recent work incorporating deep
learning see for example [12], [33], [37], and [46].
See [19] for a detailed review on reinforcement
learning with applications in ﬁnance and economics, and the references therein. However, all these
works focus on the model-free setting without taking advantage of even weak modelling assumptions
on the market dynamics. In addition, the performances of these proposed algorithms are validated
only through empirical studies and no theoretical guarantee of convergence is provided.
Organization and Notation.
For any matrix Z = (Z1, · · · , Zd) ∈Rm×d with Zj ∈Rm (j =
1, 2, · · · , d), Z⊤∈Rm×d denote the transpose of Z, ∥Z∥denotes the spectral norm of a matrix Z;
Tr(Z) denotes the trace of a square matrix Z; σmin(Z) denotes the minimal singular value of a square
matrix Z; and vec(Z) = (Z⊤
1 , · · · , Z⊤
d )⊤denote the vectorized version of a matrix Z. For a sequence
of matrices D = (D0, · · · , DT ), we deﬁne a new norm |||D||| as:
|||D||| =
T
X
t=0
∥Dt∥,
where Dt ∈Rm×d.
Further denote N(µ, Σ) as the Gaussian distribution with mean µ ∈Rd and
covariance matrix Σ ∈Rd×d.
The rest of the paper is organized as follows.
We introduce the mathematical framework and
problem set-up in Section 2. The ﬁrst step in our convergence analysis of the policy gradient method
is to consider the case of known model parameters in Section 3. When parameters are unknown, the
convergence results for the sample-based policy gradient method and projected policy gradient method
are obtained in Section 4. Finally, the algorithm is applied to liquidation problem. See Sections 2.1
and 5 for the corresponding set-up and algorithm performance, respectively.
## 2 Problem Set-up
We consider the following LQR problem over a ﬁnite time horizon T,
min
{ut}T −1
t=0
E
"T−1
X
t=0

x⊤
t Qtxt + u⊤
t Rtut

+ x⊤
T QT xT
#
,
(2.1)
such that for t = 0, 1, · · · , T −1,
xt+1 = Axt + But + wt, x0 ∼D.
(2.2)
Here xt ∈Rd is the state of the system with the initial state x0 drawn from a distribution D, ut ∈Rk
is the control at time t and {wt}T−1
t=0 are zero-mean IID noises which are independent from x0. At
this moment, we only assume x0 and {wt}T−1
t=0 have ﬁnite second moments. That is, E[x0x⊤
0 ] and
W := E[wtw⊤
t ] (∀t = 0, 1, · · · , T−1) exist. The system parameters A ∈Rd×d and B ∈Rd×k are referred
to as system (transition) matrices; Qt ∈Rd×d (∀t = 0, 1, · · · , T) and Rt ∈Rk×k (∀t = 0, 1, · · · , T −1)
are matrices that parameterize the quadratic costs. Note that the expectation in (2.1) is taken with

respect to both x0 ∼D and wt (t = 0, 1, · · · , T −1). We further denote by u := (u0, · · · , uT−1),
x := (x0, · · · , xT ), w := (w0, · · · , wT−1), Q := (Q0, · · · , QT ), and R := (R0, · · · , RT−1), the proﬁle
over the decision period T.
To solve the LQR problem (2.1)-(2.2), let us start with some conditions on the model parameters
to assure the well-deﬁnedness of the problem.
Assumption 2.1 (Cost Parameter). Assume Qt ∈Rd×d, for t = 0, 1, · · · , T, and Rt ∈Rk×k, for
t = 0, 1, · · · , T −1, are positive deﬁnite matrices.
Under Assumption 2.1, we can properly deﬁne a sequence of matrices {P ∗
t }T
t=0 as the solution to
the following dynamic Riccati equation [13]:
P ∗
t = Qt + A⊤P ∗
t+1A −A⊤P ∗
t+1B

B⊤P ∗
t+1B + Rt
−1
B⊤P ∗
t+1A,
(2.3)
with terminal condition
P ∗
T = QT .
The matrices {P ∗
t }T
t=0 can be found by solving the Riccati equations iteratively backwards in time. In
particular with a slight modiﬁcation of the initial state distribution in [13, Chapter 4.1], we have the
following result.
Lemma 2.2 (Well-deﬁnedness and the Optimal Solution [13]). Under Assumption 2.1,
1. The solution P ∗
t to the Riccati equation (2.3) is positive deﬁnite, ∀t = 0, 1, · · · , T;
2. Then the optimal control sequence {ut}T−1
t=0 is given by
ut = −K∗
t xt,
(2.4)
where
K∗
t =

B⊤P ∗
t+1B + Rt
−1
B⊤P ∗
t+1A.
(2.5)
To ﬁnd the optimal solution in the linear feedback form (2.4), we only need to focus on the following
class of linear admissible policies in feedback form
ut = −Ktxt,
t = 0, 1, · · · , T −1,
(2.6)
which can be fully characterized by K := (K0, · · · , KT−1).
2.1
Application: The Optimal Liquidation Problem
One application of the LQR framework (2.1)-(2.2) is the optimal liquidation problem. We give a slight
variant of the setup of Almgren-Chriss [8]. Our aim is to liquidate an amount q0 of an asset, with
price S0 at time 0, over the time period [0, T] with trading decisions made at discrete time points
t = 0, 1, . . . , T −1. At each time t our decision is to liquidate an amount ut of the asset. Any residual
holding is then liquidated at time T.
This will have two types of price impact.
There will be a
temporary price impact, caused when the order ‘walks the book’ and a permanent price impact as
traders rearrange their positions in the light of the sell order. We will assume the impacts are linear
in the number of traded shares.
We write St for the asset price at time t. This evolves according to a Bachelier model with a linear
permanent price impact in that
St+1 = St + σZt+1 −γut,

where, for each t = 1, . . . , T, Zt is an independent standard normal random variable, σ is the volatility
and γ is the permanent price impact parameter. The inventory process qt records the current holding
in the asset at time t. Thus we have
qt+1 = qt −ut.
Therefore, the two-dimensional state process is
St+1
qt+1

=
1
 St
qt

+
−γ
−1

ut +
σZt+1

.
(2.7)
When selling shares we incur a temporary price impact, parameter β, in that if, at time t, we trade
ut of our asset then we obtain ˜St = St−βut per share. Therefore the total revenue is PT−1
t=0 ut ˜St+qT ˜ST ,
and CT , the total cost of execution over [0, T], is the book value at time 0 minus the revenue:
CT = q0S0 −
T−1
X
t=0
ut ˜St −qT ˜ST .
In a similar way to [8], after summation by parts, we have
CT = −σ
T
X
t=1
qtZt −γ
T−1
X
t=0
u2
t + γ
 q2
0 −q2
T

+ β
T−1
X
t=0
u2
t + βq2
T .
The mean and variance of the total cost of execution are given by
E(C) =
T−1
X
t=0
δu2
t + δq2
T + γ
2q2
0,
var(C) =
T
X
t=1
σ2q2
t .
where δ = β −γ/2 summarizes the impact and is assumed positive.
Following Almgren-Chriss [8], we minimize the following cost function
CAC = min (E(C) + φ var(C)) ,
(2.8)
where φ is a parameter balancing risk versus return. For our LQR framework we take the cost function
to be
CLQR(ϵ)
=
min

E(C) + φ var(C) + ϵ
T
X
t=0
S2
t
!
=
min
T−1
X
t=0
δu2
t + δq2
T + γ
2q2
0 + φ
T
X
t=1
σ2q2
t + ϵ
T
X
t=0
S2
t
!
.
(2.9)
Note that the term ϵ PT
t=0 S2
t , with some small ϵ > 0, serves as a regularization term to guarantee
Assumption 2.1 holds. In practice, we can show that the optimal solution with ϵ small is close to the
Almgren-Chriss solution (when ϵ = 0). In addition, the algorithm will still converge with ϵ = 0. See
more discussion in Section 5. Thus, in the LQR formulation we have A =
1

, B = (−γ, −1)⊤, and
wt = (σZt+1, 0)⊤and the objective function has QT =
ϵ
δ + φσ2

, Qt =
ϵ
φσ2

and Rt = δ.
It is easy to see that Qt, for t = 0, 1, · · · , T and Rt for t = 0, 1, · · · , T −1 are positive deﬁnite, hence
Assumption 2.1 is satisﬁed.
We will show that the problem is well-deﬁned and can be solved using the methods of this paper
with rigorous convergence guarantees.

## 3 Exact Gradient Methods with Known Parameters
In this section we assume all the parameters in the model, {Qt}T
t=0, {Rt}T−1
t=0 , A, B, are known.
The analysis of exact gradient methods with known parameters paves the way for learning LQR with
unknown parameters in Section 4. In addition, numerically solving the Riccati equation (2.3) with
known parameters in high dimensions may suﬀer from computational inaccuracy [6, 41]. The exact
gradient provides a direct way of searching for the optimal solution in this case, which may be of
separate interest. Since an admissible policy can be fully characterized by K, the cost of a policy K
can be correspondingly deﬁned as
C(K) = E
"T−1
X
t=0

x⊤
t Qtxt + u⊤
t Rtut

+ x⊤
T QT xT
#
,
(3.1)
where {xt}T
t=1 and {ut}T−1
t=0
are the dynamics and controls induced by following K, starting with
x0 ∼D. Recall that K ∗is the optimal policy for the problem, in that
K ∗= arg min
K
C(K),
(3.2)
subject to the dynamics (2.2).
Well-deﬁnedness of the State Process.
To prove the global convergence of policy gradient meth-
ods, the essential idea is to show the gradient dominance condition, which states that C(K) −C(K ∗)
can be bounded by ∥∇C(K)∥F for any admissible policy K. One of the key steps to guarantee this
gradient dominance condition is the well-deﬁnedness of the state covariance matrix. That is, E[xtx⊤
t ] is
positive deﬁnite for t = 0, 1, · · · , T. This condition holds almost for free for LQR problems with inﬁnite
time horizon and deterministic dynamics. The only condition needed there is the positive deﬁniteness
of E[x0x⊤
0 ] (See [24]). However, some eﬀort needs to be made to ensure that the state covariance
matrix is well-deﬁned for LQR problems with ﬁnite horizon and stochastic dynamics. We show that
this condition holds under moderate conditions.
Assumption 3.1 (Initial State and Noise Process). We assume that
1. Initial state: x0 ∼D such that E[x0x⊤
0 ] is positive deﬁnite;
2. Noise: {wt}T−1
t=0 are IID and independent from x0 such that E[wt] = 0, and W = E[wtw⊤
t ] is
positive deﬁnite, ∀t = 0, 1, · · · , T −1.
Deﬁne σX as the lower bound over all the minimum singular values of E[xtx⊤
t ]:
σX = min
t
σmin(E[xtx⊤
t ]),
(3.3)
then we have the following result and the proof can be found in Appendix B.1.
Lemma 3.2 (Well-deﬁnedness of the State Covariance Matrix). Under Assumption 3.1, we have
E[xtx⊤
t ] is positive deﬁnite for t = 0, 1, · · · , T under any control policy K. Therefore, σX > 0.
Lemma 3.2 implies that if the initial state and the noise driving the dynamics are non-degenerate,
the covariance matrices of the state dynamics are positive deﬁnite for any policy K. However, the
covariance matrix may be degenerate in many applications, especially when inventory processes are
involved. (See, for example, the liquidation problem (2.7).) In this case, some problem-dependent
conditions are needed to guarantee that σX > 0 holds. See more discussion on the condition σX > 0

for the liquidation problem in Section 5.1. In the light of this we will assume σX > 0 in the analysis
of the convergence of the algorithm in Sections 3 and 4.
Similarly, we deﬁne σR and σQ to be the smallest values of all the minimum singular values of R
and Q:
σR = min
t
σmin(Rt),
(3.4)
and
σQ = min
t
σmin(Qt).
(3.5)
Under Assumption 2.1, we have σR > 0 and σQ > 0.
We write H = {h | h are polynomials in the model parameters} and H(.) when there are other de-
pendencies. The model parameters are in terms of d, k,
∥A∥,
∥A∥+1, ∥A∥,
∥B∥,
∥B∥+1, ∥B∥,
|||R|||,
|||R|||+1,
|||R|||,
∥W∥,
∥W∥+1, ∥W∥,
σQ ,
σQ +1, σQ,
σR ,
σR +1, σR,
σX ,
σX +1, σX, |||Q|||, E[x0x⊤
0 ], and
E[x0x⊤
0 ].
Exact Gradient Descent.
We consider the following exact gradient descent updating rule to ﬁnd
the optimal solution (3.2),
Kn+1
t
= Kn
t −η∇tC(K n), ∀0 ≤t ≤T −1,
(3.6)
where n is the number of iterations, ∇tC(K) = ∂C(K)
∂Kt
is the gradient of C(K) with respect to Kt, and
η is the step size. We further denote ∇C(K) = (∇0C(K), · · · , ∇T−1C(K)).
Let us deﬁne the state covariance matrix
Σt = E
h
xtx⊤
t
i
, t = 0, 1, · · · , T,
(3.7)
where {xt}T
t=1 is a state trajectory generated by K. Further deﬁne a matrix ΣK as the sum of Σt,
ΣK =
T
X
t=0
Σt = E
h
T
X
t=0
xtx⊤
t
i
.
(3.8)
Then, the main result for this setting is the following.
Theorem 3.3 (Global Convergence of Gradient Methods). Assume Assumption 2.1 holds. Further
assume σX > 0 and C(K 0) is ﬁnite.
Then, for an appropriate (constant) setting of the stepsize
η ∈H(
C(K 0)+1), and for ϵ > 0, if we have
N ≥
∥ΣK ∗∥
2η σX 2 σR
log C(K 0) −C(K ∗)
ϵ
,
the exact gradient descent method (3.6) enjoys the following performance bound:
C(K N) −C(K ∗) < ϵ.
The proof of Theorem 3.3 relies on the regularity of the LQR problem, some properties of the
gradient descent dynamics, and the perturbation analysis of the covariance matrix of the controlled
dynamics.

3.1
Regularity of the LQR Problem and Properties of the Gradient Descent Dy-
namics.
Let us start with the analysis of some properties of the LQR problem (2.1)-(2.2). To start, Proposition
3.4 focuses on the well-deﬁnedness of the Ricatti system {PK
t }T
t=0 induced by a control K; Lemma 3.5
gives a representation of the gradient term; Lemma 3.6 and Lemma 3.7 provide the gradient dominance
condition and a smoothness condition on the cost function C(K) with respect to policy K, respectively;
and ﬁnally, Lemma 3.8 gives two useful upper bounds on Ricatti system and state covariance matrices.
In the ﬁnite time horizon setting, deﬁne PK
t
as the solution to
PK
t
= Qt + K⊤
t RtKt + (A −BKt)⊤PK
t+1 (A −BKt) ,
t = 0, 1, · · · , T −1,
(3.9)
with terminal condition
PK
T = QT .
Note that (3.9) is equivalent to the Riccati equation (2.3) with optimal Kt = K∗
t as given by (2.5). We
have the following result on the well-deﬁnedness of PK
t
and the proof can be found in Appendix B.1.
Proposition 3.4. Under Assumption 2.1, the matrices PK
t
for t = 0, 1, . . . , T derived from (3.9) are
positive deﬁnite.
To ease the exposition, we write PK
t
as Pt when there is no confusion. Then the cost of K can be
rewritten as
C(K) = Ex0∼D
h
x⊤
0 P0x0 + L0
i
,
where, for t = 0, 1, · · · , T −1,
Lt = Lt+1 + E[w⊤
t Pt+1wt] = Lt+1 + Tr(WPt+1),
(3.10)
with LT = 0. To see this,
E[x⊤
0 P0x0] + L0 = E
"
x⊤
0 Q0x0 + x⊤
0 K⊤
0 R0K0x0 + x⊤
0 (A −BK0)⊤P1 (A −BK0) x0 +
T−1
X
t=0
w⊤
t Pt+1wt
#
= E
"
x⊤
0 Q0x0 + u⊤
0 R0u0 + x⊤
1 P1x1 +
T−1
X
t=1
w⊤
t Pt+1wt
#
= E
h T−1
X
t=0

x⊤
t Qtxt + u⊤
t Rtut

+ x⊤
T QT xT
i
.
In addition, deﬁne
Et = (Rt + B⊤Pt+1B)Kt −B⊤Pt+1A, t = 0, 1, · · · , T −1.
(3.11)
Then we have the following representation of the gradient term.
Lemma 3.5. The policy gradient has the following representation, for t = 0, 1, · · · , T −1,
∇tC(K) = 2

Rt + B⊤Pt+1B

Kt −B⊤Pt+1A

E
h
xtx⊤
t
i
= 2EtΣt.
Proof. Since
C(K) = E
h
x⊤
0 P0x0 + L0
i
= E
h
x⊤
0 (Q0 + K⊤
0 R0K0)x0 + x⊤
0 (A −BK0)⊤P1(A −BK0)x0 +
T−1
X
t=0
w⊤
t Pt+1wt

,

we have
∇0C(K) = ∂C(K)
∂K0
= E
h
2R0K0x0x⊤
0 −2B⊤P1(A −BK0)x0x⊤
i
= 2E0E
h
x0x⊤
i
= 2E0Σ0.
Similarly, ∀t = 0, 1, · · · , T −1,
∇tC(K) = 2

Rt + B⊤Pt+1B

Kt −B⊤Pt+1A

E[xtx⊤
t ] = 2EtE
h
xtx⊤
t
i
= 2EtΣt,
where the expectation E is taken with respect to both initial distribution x0 ∼D and noises w.
In classical optimization theory [24], gradient domination and smoothness of the objective function
are two key conditions to guarantee the global convergence of the gradient descent methods. To prove
that C(K) is gradient dominated, we ﬁrst prove Lemma 3.6, which indicates that for a policy K, the
distance between C(K) and the optimal cost C(K ∗) is bounded by the sum of the magnitude of the
gradient ∇tC(K) for t = 0, 1, · · · , T −1.
Lemma 3.6. Assume Assumption 2.1 holds and σX > 0. Let K ∗be an optimal policy and C(K) be
ﬁnite, then
σX
T−1
X
t=0
∥Rt + B⊤Pt+1B∥Tr(E⊤
t Et) ≤C(K) −C(K ∗) ≤
∥ΣK ∗∥
4 σX 2 σR
T−1
X
t=0
Tr(∇tC(K)⊤∇tC(K)),
where σX and σQ are deﬁned in (3.3) and (3.4).
We defer the proof of Lemma 3.6 to Appendix B.1. Lemma 3.6 implies that when the gradient
becomes small, the value of the objective function is close to C(K ∗). Now we consider the smoothness
condition of the objective function. Recall that a function f : Rn →R is said to be smooth if
|f(x) −f(y) −∇f(y)⊤(x −y)| ≤M
2 ∥x −y∥2, ∀x, y ∈Rn,
for some ﬁnite constant M. In general, it is diﬃcult to characterize the smoothness of C(K), since it
may blow up when A −BKt is large. Here we will prove that C(K) is “almost” smooth, in the sense
that when K ′ is suﬃciently close to K, C(K ′) −C(K) is bounded by the sum of the ﬁrst and second
order terms in K −K ′.
Lemma 3.7 (“Almost Smoothness”). Let {x′
t} be the sequence of states for a single trajectory generated
by K ′ starting from x′
0 = x0. Then, C(K) satisﬁes
C(K ′)−C(K) =
T−1
X
t=0
h
2 Tr

Σ′
t(K′
t−Kt)⊤Et

+Tr

Σ′
t(K′
t−Kt)⊤(Rt+B⊤Pt+1B)(K′
t−Kt)
i
, (3.12)
where Σ′
t = E

x′
t(x′
t)⊤
.
We defer the proof of Lemma 3.7 to Appendix B.1.
To see why Lemma 3.7 is related to the
smoothness, observe that when K ′ is suﬃciently close to K, in the sense that
Σ′
t ≈Σt + O(∥Kt −K′
t∥), ∀t = 0, 1, · · · , T −1,
the ﬁrst term in (3.12) will behave as Tr ((Kt −K′
t)∇tC(K)) by Lemma 3.5, and the second term in
(3.12) will be of second order in Kt −K′
t.
To utilize Lemmas 3.6 and 3.7 in the proof of Theorem 3.3, we need to further bound Pt and ΣK,
which is provided below in Lemma 3.8. The proof can be found in Appendix B.1.
Lemma 3.8. Assume Assumption 2.1 holds, and σX > 0. Then we have
∥Pt∥≤C(K)
σX
, ∥ΣK∥≤C(K)
σQ
,
where σX and σQ are deﬁned in (3.3) and (3.5).

3.2
Perturbation Analysis of ΣK.
First, let us deﬁne two linear operators on symmetric matrices. For X ∈Rd×d we set
FKt(X) = (A −BKt)X(A −BKt)⊤,
and
TK(X) := X +
T−1
X
t=0
Πt
i=0(A −BKi) X Πt
i=0(A −BKt−i)⊤.
If we write Gt = FKt ◦FKt−1 ◦· · · ◦FK0, then
Gt(X) = FKt ◦Gt−1(X) = Πt
i=0(A −BKi) X Πt
i=0(A −BKt−i)⊤,
(3.13)
and
TK(X) = X +
T−1
X
t=0
Gt(X).
(3.14)
We ﬁrst show the relationship between the operator TK and the quantity ΣK. The proof can be found
in Appendix B.1.
Proposition 3.9. For T ≥2, we have that
ΣK = TK(Σ0) + ∆(K, W),
(3.15)
where
∆(K, W) =
T−1
X
t=1
t
X
s=1
Dt,sWD⊤
t,s + T W,
with Dt,s = Πt
u=s(A −BKu) (for s = 1, 2, · · · , t), and Σ0 = E

x0x⊤

.
Let
ρ := max
n
max
0≤t≤T−1 ∥A −BKt∥,
max
0≤t≤T−1 ∥A −BK′
t∥, 1 + ξ
o
,
(3.16)
for some small constant ξ > 0. Then we have the following result on perturbations of ΣK.
Lemma 3.10 (Perturbation Analysis of ΣK). Assume Assumption 2.1 holds. Then

ΣK −ΣK ′

≤

(TK −TK ′)(Σ0)

+

∆(K, W) −∆(K ′, W)

≤ρ2T −1
ρ2 −1
C(K)
σQ
+ T∥W∥
 
2ρ ∥B∥

K −K ′

+ ∥B∥2

K −K ′

2
.
Remark 3.11. By the deﬁnition of ρ in (3.16), we have ρ ≥1 + ξ > 1. This regularization term
1 + ξ is deﬁned for ease of exposition.
Alternatively, if we deﬁne ρ := max
n
max0≤t≤T−1 ∥A −
BKt∥, max0≤t≤T−1 ∥A −BK′
t∥
o
, a similar analysis can still be carried out by considering the diﬀerent
cases: ρ < 1, ρ = 1 and ρ > 1.
The proof of Lemma 3.10 is based on the following Lemmas 3.12 and 3.13, which establish the Lipschitz
property for the operators FKt and Gt, respectively.
Lemma 3.12. It holds that, ∀t = 0, 1, · · · , T −1,
∥FKt −FK′
t∥≤2∥A −BKt∥∥B∥∥Kt −K′
t∥+ ∥B∥2∥Kt −K′
t∥2.
(3.17)

We refer to [24, Lemma 19] for the proof of Lemma 3.12.
Recall the deﬁnition of Gt in (3.13) associated with K, similarly let us deﬁne G′
t = FK′
t ◦FK′
t−1 ◦
· · · ◦FK′
0 for policy K ′. Then we have the following perturbation analysis for Gt.
Lemma 3.13 (Perturbation Analysis for Gt). For any symmetric matrix Σ ∈Rd×d, we have that
T−1
X
t=0

(Gt −G′
t)(Σ)

≤ρ2T −1
ρ2 −1
 T−1
X
t=0
∥FKt −FK′
t∥

∥Σ∥.
(3.18)
We defer the proof of Lemma 3.13 to Appendix B.2. The following perturbation analysis on T
follows immediately from Lemma 3.13.
Corollary 3.14. For any symmetric matrix Σ ∈Rd×d, we have

(TK −TK ′)(Σ)

≤ρ2T −1
ρ2 −1
 T−1
X
t=0
∥FKt −FK′
t∥

∥Σ∥,
(3.19)
where ρ is deﬁned in (3.16).
Now we are ready for the proof of Lemma 3.10.
Proof of Lemma 3.10. Using Lemma 3.12,
T−1
X
t=0
∥FKt −FK′
t∥=
T−1
X
t=0

2∥A −BKt∥∥B∥∥Kt −K′
t∥+ ∥B∥2∥Kt −K′
t∥2
≤2ρ∥B∥
T−1
X
t=0
∥Kt −K′
t∥+ ∥B∥2
T−1
X
t=0
∥Kt −K′
t∥2.
In the same way as for the proof of Lemma 3.13, we have, ∀t = 1, · · · , T −1,
t
X
s=1

Dt,sWD⊤
t,s −D′
t,sW(D′
t,s)⊤

≤ρ2T −1
ρ2 −1
t
X
s=0
∥FKs −FK′s∥
!
∥W∥.
(3.20)
By Proposition 3.9, Corollary 3.14, (3.14) and (3.20), we have

ΣK −ΣK ′

≤

(TK −TK ′)(Σ0)

+
T−1
X
t=1
t
X
s=1

Dt,sWD⊤
t,s −D′
t,sW(D′
t,s)⊤

≤ρ2T −1
ρ2 −1
 T−1
X
t=0
∥FKt −FK′
t∥

(∥Σ0∥+ T∥W∥)
≤ρ2T −1
ρ2 −1
C(K)
σQ
+ T∥W∥
 
2ρ ∥B∥

K −K ′

+ ∥B∥2

K −K ′

2
.
(3.21)
The last inequality holds since ∥Σ0∥≤∥ΣK∥≤C(K)
σQ
by Lemma 3.8.

3.3
Convergence and Complexity Analysis.
We now provide the proof of Theorem 3.3 after two preliminary Lemmas.
Lemma 3.15. Assume Assumption 2.1 holds, σX > 0, and that
K′
t = Kt −η∇tC(K),
(3.22)
where
η ≤min
(
(ρ2 −1) σQ σX
2T(ρ2T −1)(2ρ + 1)(C(K) + σQ T∥W∥)∥B∥maxt{∥∇tC(K)∥},
2C1
)
,
(3.23)
with
C1 =
C(K)
σQ
+ T∥W∥

(2ρ + 1)∥B∥(ρ2T −1)
(ρ2 −1) σX
T−1
X
t=0
∥∇tC(K)∥
!
+ 2C(K)
σQ
T−1
X
t=0
∥Rt + B⊤Pt+1B∥.
(3.24)
Then we have
C(K ′) −C(K ∗) ≤

1 −2η σR
σX
∥ΣK ∗∥

C(K) −C(K ∗)

.
We defer the proof of Lemma 3.15 to Appendix B.3.
Lemma 3.16. Assume Assumption 2.1 holds and σX > 0. Then we have that
T−1
X
t=0
∥∇tC(K)∥2 ≤4
C(K)
σQ
2 maxt ∥Rt + B⊤Pt+1B∥
σX
(C(K) −C(K ∗)),
and that:
T−1
X
t=0
∥Kt∥≤1
σR
s
T · maxt ∥Rt + B⊤Pt+1B∥
σX
(C(K) −C(K ∗)) +
T−1
X
t=0
∥B⊤Pt+1A∥

.
Proof. Using Lemma 3.8 we have
T−1
X
t=0
∥∇tC(K)∥2 ≤4
T−1
X
t=0
Tr(ΣtE⊤
t EtΣt) ≤4
T−1
X
t=0
∥Σt∥2 Tr(E⊤
t Et) ≤4
C(K)
σQ
2 T−1
X
t=0
Tr(E⊤
t Et).
From Lemma 3.6 we have
C(K) −C(K ∗) ≥σX
T−1
X
t=0
∥Rt + B⊤Pt+1B∥Tr(E⊤
t Et) ≥
σX
maxt ∥Rt + B⊤Pt+1B∥
T−1
X
t=0
Tr(E⊤
t Et),
(3.25)
and hence
T−1
X
t=0
∥∇tC(K)∥2 ≤4
C(K)
σQ
2 maxt ∥Rt + B⊤Pt+1B∥
σX
(C(K) −C(K ∗)).

For the second claim, using Lemma 3.6 again,
T−1
X
t=0
∥Kt∥=
T−1
X
t=0
∥(Rt + B⊤Pt+1B)−1Kt(Rt + B⊤Pt+1B)∥
≤
T−1
X
t=0
σmin(Rt)∥Kt(Rt + B⊤Pt+1B)∥≤
T−1
X
t=0
σmin(Rt)

∥Et∥+ ∥B⊤Pt+1A∥

≤
T−1
X
t=0


q
Tr(E⊤
t Et)
σmin(Rt)
+ ∥B⊤Pt+1A∥
σmin(Rt)

≤1
σR

v
u
u
tT ·
T−1
X
t=0
Tr(E⊤
t Et) +
T−1
X
t=0
∥B⊤Pt+1A∥

≤1
σR
s
T · maxt ∥Rt + B⊤Pt+1B∥
σX
(C(K) −C(K ∗)) +
T−1
X
t=0
∥B⊤Pt+1A∥

.
The second inequality holds by the deﬁnition of Et in (3.11), the second last step uses the Cauchy-
Schwarz inequality, and the last inequality holds by (3.25).
Proof of Theorem 3.3. In order to show the existence of a positive η such that (3.23) holds, it suﬃces to
show there exists a positive lower bound on the RHS of (3.23). By Lemma 3.16 and the Cauchy-Schwarz
inequality,
T−1
X
t=0
∥∇tC(K)∥≤
v
u
u
tT ·
T−1
X
t=0
∥∇tC(K)∥2
≤
s
4T ·
C(K)
σQ
2 maxt ∥Rt + B⊤Pt+1B∥
σX
(C(K) −C(K ∗)).
(3.26)
Note that if d < ab + c for some a > 0, b > 0 c > 0 and d > 0, then 1
d >
(a+1)(b+1)(c+1). Also
an+1 >
(a+1)n for a > 0 and n ∈N+. Therefore, based on (3.24) and (3.26),
C1 is bounded below by
polynomials in 1
ρ,
C(K)+1,
∥B∥+1,
|||R|||+1,
∥W∥+1, σX, σQ,
σX +1, and
σQ +1.
Now we aim to show that 1
ρ is bounded below by some polynomials in the parameters. To see this,
let us ﬁrst show that ρ is bounded above by polynomials in ∥A∥, ∥B∥, |||R|||,
σX ,
σR and C(K). Since
∥B∥∥K′
t −Kt∥≤
σQ σX
4C(K) ≤1
2 holds under the assumptions in Lemma 3.15, we have
max
0≤t≤T−1 ∥A −BK′
t∥≤
max
0≤t≤T−1
 ∥A −BKt∥+ ∥B∥∥K′
t −Kt∥

≤
max
0≤t≤T−1 ∥A −BKt∥+ 1
2,
thus
ρ = max
n
max
0≤t≤T−1 ∥A −BKt∥,
max
0≤t≤T−1 ∥A −BK′
t∥, 1 + ξ
o
≤max
n
max
0≤t≤T−1 ∥A −BKt∥+ 1
2, 1 + ξ
o
≤max
n
∥A∥+ ∥B∥
T−1
X
t=0
∥Kt∥+ 1
2, 1 + ξ
o
.
(3.27)
Given the bound on PT−1
t=0 ∥Kt∥by Lemma 3.16 and ∥Pt∥≤C(K)
σX
by Lemma 3.8, ρ is bounded above
by polynomials in ∥A∥, ∥B∥, |||R|||,
σX ,
σR and C(K), or a constant 1+ξ. Therefore 1
ρ is bounded below
by polynomials in
∥A∥+1,
∥B∥+1,
|||R|||+1, σX, σR and
C(K)+1, or a constant
1+ξ. Hence, by choosing

η ∈H(
C(K 0)+1) to be an appropriate polynomial in
C(K 0),
C(K 0)+1,
∥A∥+1,
∥B∥+1,
|||R|||+1,
∥W∥+1, σX,
σQ, σR,
σX +1, and
σQ +1, (3.23) is satisﬁed, since by performing gradient descent, C(K 1) < C(K 0).
Therefore, by Lemma 3.15, we have
C(K 1) −C(K ∗) ≤

1 −2η σR
σX
∥ΣK ∗∥

C(K 0) −C(K ∗)

,
which implies that the cost decreases at t = 1. Suppose that C(K n) ≤C(K 0), then the stepsize
condition in (3.23) is still satisﬁed by Lemma 3.16. Thus, Lemma 3.15 can again be applied for the
update at round n + 1 to obtain:
C(K n+1) −C(K ∗) ≤

1 −2η σR
σX
∥ΣK ∗∥

C(K n) −C(K ∗)

.
For ϵ > 0, provided N ≥
∥ΣK∗∥
2η σX 2 σR log C(K 0)−C(K ∗)
ϵ
, we have
C(K N) −C(K ∗) < ϵ.
## 4 Sample-based Policy Gradient Method with Unknown Parameters
In the setting with unknown parameters, the controller has only simulation access to the model; the
model parameters, A, B, {Qt}T
t=0, {Rt}T−1
t=0 , are unknown.
By using a zeroth-order optimization
method to approximate the gradient, this section proves the policy gradient method with unknown
parameters also leads to a global optimal policy, with both polynomial computational and sample
complexities.
Note that in this section, when bounding the Frobenius norm of a matrix, we usually treat the
matrix as a stacked vector. Therefore we denote by D = k×d the dimension of the corresponding vector
formed from the K matrix for convenience in the proofs. Therefore in each iteration n = 1, 2, · · · , N,
we can update the policy as, for t = 0, 1, · · · , T −1,
Kn+1
t
= Kn
t −η
\
∇tC(K n),
(4.1)
where
\
∇tC(K n) is the estimate of ∇tC(K n). We analyze the following Algorithm 1.
Algorithm 1 Policy Gradient Estimation with Unknown Parameters
1: Input: K, number of trajectories m, smoothing parameter r, dimension D
2: for i ∈{1, . . . , m} do
3:
for t ∈{0, . . . , T −1} do
4:
Sample the (sub)-policy at time t: bKi
t = Kt +Ui
t where Ui
t is drawn uniformly at random over
matrices such that ∥Ui
t∥F = r.
5:
Denote bcti as the single trajectory cost with policy
(K −t, bKi
t) := (K0, · · · , Kt−1, bKi
t, Kt, · · · , KT−1) starting from xi
0 ∼D.
6:
end for
7: end for
8: Return the estimates of ∇tC(K) for each t:
\
∇tC(K) = 1
m
m
X
i=1
D
r2 bcti Ui
t.
(4.2)

Remark 4.1. [Zeroth-order Optimization Approach in the Sub-routine (4.2)] In the estimation of
the gradient term (4.2), we adopt a zeroth-order optimization method, using only query access to a
sample of the reward function c(·) at input points K, without querying the gradients and higher order
derivatives of c(·). In a similar way to the observation in [24], the objective C(K) may not be ﬁnite
for every policy K when Gaussian smoothing is applied, therefore EU ∼N(0,σ2I)[C(K + U )] may not be
well-deﬁned. This is avoidable by smoothing over the surface of a ball. The step (4.2) (in Algorithm
1) provides a procedure to ﬁnd an (bounded bias) estimate ∇\
C(K) of ∇C(K).
To guarantee the global convergence of the sample-based algorithm (Algorithm 1), we propose some
conditions on the distribution of x0 and {wt}T−1
t=0 , in addition to the ﬁnite second moment condition
speciﬁed in Section 2.
Deﬁnition 4.2. A zero-mean random variable X
1. is said to be sub-Gaussian with variance proxy σ2 and we write X ∈SG(σ2) if its moment
generating function satisﬁes E[exp(λX)] ≤exp

λ2σ2

.
2. is said to be sub-exponential with parameters (ν2, α) and we write X ∈SE(ν2, α), if E[exp(λX)] ≤
exp

λ2ν2

for any λ such that |λ| ≤1
α.
We assume the initial distribution and the noise in the state process dynamics satisfy the following
assumptions.
Assumption 4.3 (Initial State and Noise Process (II)).
1. Initial state: x0 = f
W0z0 where
z0 = (z0,1 · · · , z0,d) ∈Rd is a random vector with independent components z0,i which are sub-
Gaussian, mean-zero, and have sub-Gaussian parameter σ2
0; f
W0 ∈Rd×d is an unknown and
deterministic matrix.
2. Noise process: wt = f
Wvt where vt := (vt,1 · · · , vt,d) ∈Rd are IID and independent from x0.
vt has independent components vt,i which are sub-Gaussian, mean-zero, and have sub-Gaussian
parameter σ2
w, ∀t = 0, 1, · · · , T −1. f
W ∈Rd×d is an unknown and deterministic matrix.
Note that Assumptions 3.1 and 4.3 serve diﬀerent purposes in this paper. Assumption 3.1 provides
one suﬃcient condition to assure σX > 0. Assumption 4.3 is used to guarantee the convergence of the
sample based algorithm (Algorithm 1).
In addition to the model parameters speciﬁed in Section 3, here we assume H(·) includes polynomials
that are also functions of σ0,
σ0 ,
σ0+1, σw,
σw ,
σw+1 ∥f
W∥,
∥f
W∥,
∥f
W∥+1, ∥f
W0∥,
∥f
W0∥, and
∥f
W0∥+1.
Theorem 4.4. Assume Assumptions 2.1 and 4.3 hold and further assume σX > 0 and C(K 0) is ﬁnite.
At every step the policy is updated as in (4.1), that is
Kn+1
t
= Kn
t −η
\
∇tC(K n),
with η ∈H(
C(K 0)+1) and
\
∇tC(K n) is computed with hyper-parameters (r, m) such that r < 1/hradius
and m > hsample with some ﬁxed polynomials hradius ∈H(1/ϵ, C(K 0)) and hsample ∈H(1/ϵ, C(K 0)).
Then for ϵ > 0, if we have
N ≥
∥ΣK ∗∥
η σX 2 σR
log C(K 0) −C(K ∗)
ϵ
,
it holds that C(K N) −C(K ∗) ≤ϵ with high probability (at least 1 −exp(−D)).
The proof of Theorem 4.4 is based on a perturbation analysis of C(K) and ∇tC(K), smoothing and
the gradient descent analysis of the procedures in Algorithm 1. We provide the perturbation analysis
and the smoothing analysis in Sections 4.1 and 4.2, respectively. We defer the proof of Theorem 4.4
to Section 4.3.

Projected Policy Gradient Method.
In many situations constrained optimization problems arise
and the projected gradient descent method is one popular approach to solve such problems. Recall the
projection of a point y ∈Rk×(T×d) onto a set SK ⊂Rk×(T×d) is deﬁned as
ΠSK (y) = arg min
x∈SK
2 ∥x −y∥2
2 .
(4.3)
Then the projected policy gradient (PPG) updating rule can be deﬁned as
K n+1 = ΠSK

K n −η \
∇C(K n)

,
(4.4)
where
\
∇C(K n) =

\
∇0C(K n), · · · ,
\
∇T−1C(K n)

denotes the estimate of ∇C(K n).
If the projection set SK is convex and closed, the projection onto SK is non-expansive, that is,
∥ΠSK (x) −ΠSK (y)∥2 ≤∥x −y∥2 . Therefore the results in Theorem 4.4 for the standard policy gradient
method can be easily generalized to the following Theorem 4.5 for the PPG version.
Theorem 4.5. Assume Assumptions 2.1 and 4.3 hold, and the projection set of policy K, denoted by
SK, is convex and closed. Further assume σX > 0 and C(K 0) is ﬁnite. At every step the policy is
updated as in (4.4), that is
K n+1 = ΠSK

K n −η \
∇C(K n)

with η ∈H(
C(K 0)+1) and
\
∇tC(K n) (t = 0, 1, · · · , T −1) is computed with hyper-parameters (r, m)
such that r < 1/bhradius and m > bhsample with some ﬁxed polynomials bhradius ∈H(1/ϵ, C(K 0)) and
bhsample ∈H(1/ϵ, C(K 0)). Then for ϵ > 0, if we have
N ≥
∥ΣK ∗∥
η σX 2 σR
log C(K 0) −C(K ∗)
ϵ
,
it holds that C(K N) −C(K ∗) ≤ϵ with high probability (at least 1 −exp(−D)).
4.1
Perturbation analysis of C(K) and ∇tC(K)
This section shows that the objective function C(K) and its gradient are stable with respect to small
perturbations. The proofs of the following Lemmas can be found in Appendix B.4.
Lemma 4.6 (C(K) Perturbation). Assume Assumptions 2.1 and 4.3 hold, σX > 0, and K ′ such that,
∀t = 0, 1, · · · , T −1,
∥K′
t −Kt∥≤min
(
(ρ2 −1) σQ σX
2T(ρ2T −1)(2ρ + 1)(C(K) + σQ T∥W∥)∥B∥, ∥Kt∥, 1
T
)
,
(4.5)
where ρ is deﬁned in (3.16). Then there exists a polynomial hcost ∈H(C(K)) such that
|C(K ′) −C(K)| ≤hcost

K ′ −K

.
Lemma 4.7 (∇tC(K) Perturbation). Under the same assumptions as in Lemma 4.6, there exists a
polynomial hgrad ∈H(C(K)) such that
∥∇tC(K ′) −∇tC(K)∥≤hgrad

K ′ −K

,
and
∥∇tC(K ′) −∇tC(K)∥F ≤hgrad

K ′ −K

F .

4.2
Smoothing and the Gradient Descent Analysis
In this section, Lemma 4.8 provides the formula for the perturbed gradient term, Lemma 4.9 provides
the concentration inequality for ﬁnite samples, and Lemma 4.10 provides the guarantees for the gradient
approximation.
Recall that D = k × d. Let Sr represent the uniform distribution over the points with norm r
in dimension D, and Br represent the uniform distribution over all points with norm at most r in
dimension D.
For each Kt (t = 0, 1, · · · , T −1), the algorithm performs gradient descent on the
following function:
Cr
t (K) = EVt∼Br [C(K + V t)] ,
(4.6)
where V t := (0, · · · , Vt, · · · , 0) and Vt ∈Rk×d.
Lemma 4.8. Assume C(K) is ﬁnite,
∇tCr
t (K) = D
r2 EUt∼Sr[C(K + U t)Ut].
(4.7)
The proof of Lemma 4.8 is similar to the proof of [24, Lemma 29] and hence omitted.
We ﬁrst state two facts on sub-Gaussian and sub-exponential random variables.
Firstly, if X
and Y are zero-mean independent random variables such that X ∈SG(σ2
x) and Y ∈SG(σ2
y), then
XY ∈SE(σxσy, 4σxσy). Secondly, if X1, · · · , Xn are zero-mean independent random variables such
that Xi ∈SE(ν2
i , αi), then
n
X
i=1
Xi ∈SE
n
X
i=1
ν2
i , max
i
αi
!
.
Utilizing above two facts, we have the following.
Lemma 4.9. Assume Assumptions 2.1 and 4.3 hold and σX > 0, then there exist polynomials ν ∈
H(C(K)) and α ∈H(C(K)) such that
"T−1
X
t=0

x⊤
t Qtxt + u⊤
t Rtut

+ x⊤
T QT xT
#
is sub-exponential with parameter
 ν2, α

. Here {xt}T
t=0 is the dynamics under policy K.
Proof. We ﬁrst observe that, by direct calculation,
"T−1
X
t=0

x⊤
t Qtxt + u⊤
t Rtut

+ x⊤
T QT xT
#
= x⊤
0 P0x0 +
T−1
X
t=0
w⊤
t Pt+1wt.
(4.8)
Note that by (3.9) and Proposition 3.4, Pt is symmetric and positive deﬁnite. The Frobenius norm
∥· ∥F and the spectral norm ∥· ∥of the matrix Pt ∈Rd×d have the following property:
∥Pt∥≤∥Pt∥F ≤
√
d∥Pt∥, ∀t = 0, 1, · · · , T.
(4.9)
Let bσ = max{σ0, σw}. Given the Hanson-Wright inequality (Theorem 2.5 in [4]),
P


w⊤
t Pt+1wt −E
h
w⊤
t Pt+1wt
i

≥t

= P


v⊤
t (f
W ⊤Pt+1f
W)vt −E
h
v⊤
t (f
W ⊤Pt+1f
W)vt
i

≥t

≤2 exp

−c min
(
t2
2bσ4∥f
W ⊤Pt+1f
W∥2
F
,
t
bσ2∥f
W ⊤Pt+1f
W∥
)!
,
(4.10)

for some universal constant c > 0 which is independent of Pt+1 and wt.
Combining (4.9), (4.10) and Lemma 3.8,
P


w⊤
t Pt+1wt −E
h
w⊤
t Pt+1wt
i

≥t

≤
2 exp

−c min
(
t2
2bσ4 d ∥Pt+1∥2∥f
W∥4 ,
t
bσ2∥Pt+1∥∥f
W∥2
)!
≤
2 exp

−c min
(
t2
2bσ4∥f
W∥4 d C2(K)/ σX 2 ,
t
bσ2∥f
W∥2C(K)/ σX
)!
.
Therefore the random variable w⊤
t Pt+1wt is sub-exponential with parameters

bσ4∥f
W∥4dC2(K)
c σX 2
, bσ2∥f
W∥2C(K)
2c σX

.
In the same way x⊤
0 P0x0 is sub-exponential with parameters

bσ4∥f
W0∥4dC2(K)
c σX 2
, bσ2∥f
W0∥2C(K)
2c σX

.
Let
σ = max{∥f
W0∥, ∥f
W∥}.
Since {wt}T−1
t=0
are IID and independent from x0, we have (4.8) is sub-
exponential with parameters

(T + 1) bσ4σ4dC2(K)
c σX 2
, bσ2σ2C(K)
2c σX

.
Deﬁne
e∇t := 1
m
m
X
i=1
D
r2 C(K + U i
t)Ui
t

as the average of perturbed cost functions across m scenarios which is an empirical approximation of
(4.7). Similarly, deﬁne
b∇t := 1
m
m
X
i=1

D
r2
"T−1
X
t=0

(xi
t)⊤Qtxi
t + (ui
t)⊤Rtui
t

+ (xi
T )⊤QT xi
T
#
Ui
t
!
(4.11)
as the average of perturbed and single-trajectory-based cost functions across m scenarios, which is the
same as (4.2) in Algorithm 1. Note that in order to calculate e∇t, we require access to C(K + U i
t),
which involves the calculation of expectations with respect to unknown initial states and state noises.
This may be restrictive in some settings.
On the other hand, the calculation of b∇t only involves
single-trajectory-based cost functions.
Lemma 4.10. Assume Assumptions 2.1 and 4.3 hold, and σX > 0. Given any ϵ, there are ﬁxed
polynomials hradius ∈H(1/ϵ, C(K)) and hsample ∈H(1/ϵ, C(K)) such that when r ≤1/hradius, with
m ≥hsample samples of U1
t , · · · , Um
t
∼Sr for each t = 0, · · · , T −1,

e∇t −∇tC(K)

F ≤ϵ,
holds with high probability (at least 1 −
  D
ϵ
−D).
In addition, there is a polynomial hsample,2 ∈
H(1/ϵ, C(K)) such that when r ≤1/hradius, with m ≥hsample + hsample,2 samples of U1
t , ..., Um
t
∼Sr
for each t = 0, · · · , T −1,

b∇t −∇tC(K)

F ≤3
2ϵ,
holds with high probability (at least 1 −2
  D
ϵ
−D). Here, for each i = 1, 2, · · · , m, {xi
t}T
t=0 and {ui
t}T−1
t=0
are the dynamics and controls for a single path sampled using policy K + U i
t.

Proof. Note that
e∇t −∇tC(K) = (∇tCr
t (K) −∇tC(K)) + (e∇t −∇tCr
t (K)),
where Cr
t is deﬁned in (4.6).
For the ﬁrst term, choose hradius = max{1/r0, 4hgrad/ϵ} (r0 is chosen later), where hgrad ∈H(C(K))
is deﬁned in Lemma 4.7. By Lemma 4.7 when r ≤1/hradius ≤ϵ/4hgrad, for V t := (0, · · · , Vt, · · · , 0)
where Vt ∼Br, we have
∥∇tC(K + V t) −∇tC(K)∥F ≤hgrad∥V t∥F ≤hgrad
ϵ
4hgrad
= ϵ
4.
(4.12)
Since ∇tCr
t (K) = EVt∼Br[∇tC(K + V t)], we have
∥∇tC(K + V t) −∇tCr
t (K)∥F ≤ϵ
4,
by (4.12) and the continuity of ∇tC. Therefore
∥∇tCr
t (K) −∇tC(K)∥F ≤∥∇tC(K + V t) −∇tC(K)∥F + ∥∇tC(K + V t) −∇tCr
t (K)∥F ≤ϵ
2 (4.13)
holds by triangle inequality. We choose r0 such that for any U t ∼Sr, we have that C(K +U t) ≤2C(K).
By Lemma 4.6, we can pick 1/r0 = hcost/C(K), then |C(K + U t) −C(K)| ≤r0 · hcost ≤C(K).
For the second term, by Lemma 4.8, E[e∇t] = ∇tCr
t (K), and each individual sample is bounded by
2DC(K)/r, so by the Operator-Bernstein inequality [28, Theorem 12] with
m ≥hsample = Θ

D
D · C(K)
rϵ
2
log(D/ϵ)
!
,
we have
P
h

e∇t −∇tCr
t (K)

F ≤ϵ
i
≥1 −
D
ϵ
−D
.
(4.14)
Note that hsample ∈H(1/ϵ, C(K)) since 1/r > hradius ∈H(1/ϵ, C(K)).
Adding these two terms
together and applying the triangle inequality gives the result.
For the second part, note that
Ex0,w[b∇t] = e∇t.
(4.15)
By Lemma 4.9,
"T−1
X
t=0

(xi
t)⊤Qtxi
t + (ui
t)⊤Rtui
t

+ (xi
T )⊤QT xi
T
#
is sub-exponential with parameters (ν2, α). Therefore,
Zi :=

D
r2
"T−1
X
t=0

(xi
t)⊤Qtxi
t + (ui
t)⊤Rtui
t

+ (xi
T )⊤QT xi
T
#
Ui
t
!
is sub-exponential matrix with parameters (eν2, eα) :=
  D
r2 ν2, α

. Then by Operator-Berinstein inequal-
ity [28, Theorem 12],
P
"

m
m
X
i=1
Zi −E[Z1]

F
≤t
#
≥1 −2D exp

−m t2
2eν2

,

when t ≤eν2
eα . That is, there exists a polynomial hsample,2 ∈H(1/ϵ, C(K)) where
hsample,2 := hsample,2

D, 1
ϵ , 1
r, σ0, σw, ∥f
W0∥, ∥f
W∥, C(K), 1
σX

= Θ

D
eν
ϵ
2
log(D/ϵ)
!
,
such that when m ≥hsample,2,
P
h

b∇t −e∇t

F ≤ϵ
i
≥1 −
D
ϵ
−D
.
(4.16)
Combining (4.16) with (4.14) and (4.13), we arrive at the desired result.
4.3
Proof of Theorem 4.4
With the results in Section 4.1 and Section 4.2, now we are ready to prove the main theorem.
Proof of Theorem 4.4. By Lemma 3.15 and by choosing η ∈H(
C(K 0)+1) such that the step size con-
dition (3.23) is satisﬁed,
C(K ′) −C(K ∗) ≤

1 −2η σR
σX
∥ΣK ∗∥

C(K) −C(K ∗)

.
Recall the deﬁnition of b∇t in (4.11) and let K′′
t = Kt −η b∇t be the iterate that uses the approximate
gradient. We will show later that given enough samples, the gradient can be estimated with enough
accuracy that makes sure
|C(K ′′) −C(K ′)| ≤η σR
σX
∥ΣK ∗∥ϵ.
(4.17)
That means as long as C(K) −C(K ∗) ≥ϵ, we have
C(K ′′) −C(K ∗) ≤

1 −η σR
σX
∥ΣK ∗∥

C(K) −C(K ∗)

.
Then the same proof as that of Theorem 3.3 gives the convergence guarantee.
Now let us prove (4.17). First note that C(K ′′)−C(K ′) is bounded. By Lemma 4.6, if ∥K′′
t −K′
t∥≤
η σR
σX
∥ΣK∗∥·ϵ/(T ·hcost), where hcost ∈H(C(K)) is the polynomial in Lemma 4.6, then (4.17) holds. To
get this bound, recall K′
t = Kt −η∇tC(K) in (3.22) and writing ∇t = ∇tC(K) for ease of exposition,
observe that K′′
t −K′
t = η(∇t −b∇t), therefore it suﬃces to make sure
∥∇t −b∇t∥≤
σX
2 σR
T∥ΣK ∗∥hcost
ϵ.
By Lemma 4.10, it is enough to pick hradius = hradius(3T∥ΣK ∗∥hcost(C(K))/(2 σX
2 σR ϵ), C(K)) ∈
H(1/ϵ, C(K)), and
hsample
=
hsample
3hcost(C(K))∥ΣK ∗∥
2 σX 2 σR ϵ
, C(K)

+ hsample,2
3hcost(C(K))∥ΣK ∗∥
2 σX 2 σR ϵ
, C(K)

.
This gives the desired upper bound on ∥∇t −b∇t∥with high probability (at least 1 −2(ϵ/D)D).
Since the number of steps is a polynomial, we have TN = o(ϵD).
By the union bound with
probability at least

1 −2
 ϵ
D
DTN
≥1 −2 TN
 ϵ
D
D
≥1 −exp(−D),

we have ∥∇t −b∇t∥≤
σX
2 σR
T∥ΣK∗∥hcost ϵ, ∀t = 0, 1, · · · , T −1. Therefore,
C(K ′′) −C(K ∗) ≤

1 −η σR
σX
∥ΣK ∗∥

C(K) −C(K ∗)

.
(4.18)
This implies C(K ′′) < C(K).
To guarantee that (4.18) holds at each iteration n = 1, 2, · · · , N,
it suﬃces to pick hradius ∈H(1/ϵ, C(K 0)) and hsample ∈H(1/ϵ, C(K 0)). The rest of the proof is
the same as that of Theorem 3.3. Note again that in the smoothing, because the function value is
monotonically decreasing, and by the choice of radius, all the function values encountered are bounded
by 2C(K 0), so the polynomials are indeed bounded throughout the algorithm.
4.4
Discussion
Remark 4.11 (Comparison with [24]). The proofs of our main results, Theorems 3.3 and 4.4, are
diﬀerent to those from [24].
Firstly, to prove the gradient dominant condition, [24] only required
conditions on the distribution of the initial position. However, we need conditions to guarantee the
non-degeneracy of the state covariance matrix at any time. Secondly, the extra randomness from the
sub-Gaussian noise needs to be taken care of in the perturbation analysis of ΣK. Finally, we need
more advanced concentration inequalities to provide the number of samples and number of simulation
trajectories that leads to the theoretical guarantee in the case with unknown parameters.
Remark 4.12 (Non-stationary Dynamics). Note that our framework can be generalized to non-
stationary dynamics, that is, for t = 0, 1, · · · , T −1,
xt+1 = Atxt + Btut + wt, x0 ∼D.
(4.19)
with {At}T−1
t=0 and {Bt}T−1
t=0 time-dependent state parameters.
Remark 4.13 (Other Policy Gradient Methods). Our convergence and sample complexity analysis
could be applied to other policy gradient methods, including the Natural policy gradient method and
the Gauss-Newton method, in the framework of the LQR with stochastic dynamics and ﬁnite horizons.
## 5 Numerical Experiments
The performance of the PPG algorithm (4.4) is demonstrated for the optimal liquidation problem with
single asset and the empirical analysis of the policy gradient method (4.1) in higher dimensions is also
provided with synthetic data. We will speciﬁcally focus on the following questions.
• In practice, how fast do the policy gradient algorithm and the PPG algorithm with known and
unknown parameters converge to the true solution?
• How does the deadline (ﬁnite horizon) inﬂuence the optimal policy?
• When the real-word system does not exactly follow the LQR framework, does policy-gradient
outperforms mis-speciﬁed LQR models?
This section is organized as follows.
We demonstrate the performance of the PPG algorithms for
optimal liquidation problem with single asset in LQR framework in Section 5.1. We then show that
without the LQR model speciﬁcation, the learned policy from the policy gradient algorithm improves
the Almgren-Chriss solution in Section 5.2. Finally, we test the performance of the algorithm with
unknown parameters in high dimensions in Section 5.3.

5.1
Optimal Liquidation within the LQR Framework
Recall the set up of the optimal liquidation problem in (2.1). By convention, we write the control
in the feedback form as ut = −Ktxt. Writing Kt = (k1
t , k2
t ), we have ut = −k1
t St −k2
t qt, the state
equation becomes
xt+1 =
1 + γk1
t
γk2
t
k1
t
1 + k2
t

xt + wt.
In the liquidation problem, we assume ut ≥0 (0 ≤t ≤T −1).
That is, k1
t ≤0 and k2
t ≤0
(0 ≤t ≤T −1).
Assumption 5.1 (Assumptions for the Optimal Liquidation Problems). We assume
(1) γk1
t + k2
t > −1 (0 ≤t ≤T −1);
(2) β > γ
2.
Justiﬁcation of the Assumption.
Assumption 5.1-(1) is essential to ensure that the liquidation
problem is well deﬁned. First, γk1
t > −1 makes sure that the stock price process {St}T
t=0 is well-
behaved:
E[St+1] = E[St] −γE[ut] = (1 + γk1
t )E[St] + γk2
t qt.
If γk1
t < −1, then E[St+1] ≤0 since k2
t ≤0. Second, k2
t ≥−1 guarantees that inventory will not be
negative. Note that
qt+1 = qt −(−k1
t St −k2
t qt) = (1 + k2
t )qt + k1
t St.
If k2
t ≤−1 and qt > 0, then qt+1 < 0. Assumption 5.1-(2) implies that the temporary market impact is
“bigger” than one half of the permanent market impact, which is consistent with the empirical evidence
[9] and assumptions in [8].
Learning to Liquidate.
In practice, traders may not know the market impact parameter γ. But
one can always take some ¯γ > γ based on some basic understandings of the market and perform a
PPG algorithm to the closed convex set SK:
SK :=

K = (K0, · · · , KT−1) : Kt = (k1
t , k2
t ), ¯γk1
t + k2
t ≥−1 + ζ, k1
t ≤0, k2
t ≤0, ∀t = 0, · · · , T −1

,
(5.1)
with some small parameter ζ > 0.
In practice γ is usually on the order of 10−5 ∼10−6 (See Table 3 in Appendix A) and hence a
universal upper bound ¯γ in (5.1) is not a strong assumption for a given portfolio of stocks to liquidate.
Proposition 5.2. Assume K ∈SK and Assumptions 2.1, 4.3 and 5.1 hold, we have σX > 0 and
{PK
t }T
t=0 derived from (3.9) are positive deﬁnite for the optimal liquidation problem (2.7) and (2.9).
The proof of Proposition 5.2 is deferred to Appendix B.5. It is easy to check that the projection
set SK deﬁned in (5.1) is convex and closed. Along with Proposition 5.2, the convergence result in
Theorem 4.5 holds for the liquidation problem (2.7) and (2.9) as long as the conditions in Proposition
5.2 are satisﬁed.
We test the performance of the PPG algorithm with projection set SK on Apple (AAPL) and
Facebook (FB) stocks. The market simulator of the associated LQR framework is constructed with
NASDAQ ITCH data and the details can be found in Appendix A.

Performance Measure.
We use the following normalized error to quantify the performance of a
given policy K,
Normalized error = C(K) −C(K ∗)
C(K ∗)
,
where K ∗is the optimal policy deﬁned in (2.5).
Set-up.
(1) Parameters: φ = 5 × 10−6 (for both AAPL and FB), ϵ = 10−8, T = 10; smoothing
parameter r = 0.6, number of trajectories m = 200; initial policy K 0 ∈R1×2T with {K 0}ij = −0.2 for
all i, j, for both algorithms with known and unknown parameters; step sizes are indicated in the ﬁgures;
¯γ = 5×10−5, ζ = 10−12 for the projection set. (2) Initialization: Assume the initial inventory q0 follows
N(500, 1). The small variance of the initial inventory distribution is used to guarantee the initial state
covariance matrix is positive deﬁnite. In practice, the algorithm converges with deterministic initial
inventories.
(a) PPG with known parameters (η = 0.1).
(b) PPG with unknown parameters (η = 0.05).
Figure 1: Performance of the PPG algorithms (50 simulation scenarios).
Convergence.
Both PPG algorithms with known parameters and unknown parameters show a rea-
sonable level of accuracy within 50 iterations (that is the normalized error is less than 10−2). The
PPG algorithm with known parameters has almost no ﬂuctuations across the 50 scenarios. By choos-
ing m = 200, the performance of the PPG algorithm with unknown parameters is stable with relatively
small ﬂuctuations (see the blue area in Figure 1b) across the 50 scenarios.
(a) AAPL.
(b) FB.
Figure 2: Optimal inventory trajectory under diﬀerent deadlines (200 simulation scenarios).

Impact of the Deadline.
The optimal policy is sensitive to the deadline in that the shapes of the
optimal inventory trajectories are diﬀerent with diﬀerent deadlines. See Figure 2 for both AAPL and
FB with T = 30, 60 and 120 minutes. The liquidation speed is almost linear when T is small; and it is
faster in the initial trading phase and slower at the end when T is relatively large.
Impact of the Parameter φ.
Recall that in (2.9) the parameter φ is used to balance the expected
terminal wealth E[C] and the variance of the terminal wealth var[C]. To show the impact of φ, we
set φ to be 10−4, 10−5, 10−6, and 10−7 and show the corresponding inventory trajectories in Figure 3.
The optimal liquidation speed is almost linear when φ is small, while it is faster in the initial trading
phase and slower at the end when φ is relatively large.
Figure 3: Inventory trajectories
of AAPL under diﬀerent φ (aver-
age across 200 simulation scenar-
ios).
(a) Inventory trajectories.
(b) Relative cost diﬀerence.
Figure 4:
Original Almgren-Chriss framework versus LQR formula-
tion under diﬀerent ϵ (AAPL).
Impact of the Parameter ϵ.
Recall that our liquidation formulation (2.9) diﬀers from the Almgren-
Chriss formulation (2.8) by an additional regularization term PT
t=0 ϵS2
t . The role of this term is to
enable the problem to be cast in the LQR framework and to guarantee the well-deﬁnedness of the
Ricatti equation.
From Figure 4a, the optimal policies and inventory trajectories are close to the
Almgren-Chriss solution when ϵ ≤0.01. However, when ϵ = 0.05, the optimal policy is far away from
the Almgren-Chriss solution. We show the diﬀerence between CAC, deﬁned in (2.8), and CLQR(ϵ),
deﬁned in (2.9), in Figure 4b. We see that CLQR(ϵ) is close to CAC when ϵ < 0.02 and is markedly
diﬀerent from CAC when ϵ ≥0.02. It is worth noticing that when ϵ = 0, the algorithm does converge
to the Almgren-Chriss solution in our setting although the convergence of the algorithm in this case is
not guaranteed by our theoretical results.
5.2
Learning to Liquidate without Model Speciﬁcation
In practice, the dynamics of the trading system may not be exactly those assumed in the LQR frame-
work but we might expect that the policy gradient method could still perform well when the system
is “nearly” linear quadratic as the execution of the policy gradient method does not rely on the model
speciﬁcation. In this section, we consider liquidation problems in the Limit Order Book (LOB) setting.
A LOB is a list of orders that a trading venue, for example the NASDAQ exchange, uses to record
the interest of buyers and sellers in a particular ﬁnancial instrument. There are two types of orders
the buyers (sellers) can submit: a limit buy (sell) order with a preferred price for a given volume or a
market buy (sell) order with a given volume which will be immediately executed with the best available
limit sell (buy) orders. Here we perform the policy gradient method to learn the optimal strategies to
liquidate using market orders in the LOB.

We denote by St the mid-price of the asset at time t, that is the average of the best-bid price and
best-ask price. At each time t, the decision is to liquidate an amount ut of the asset. The action ut
will have an impact on the market, with possibly both temporary and permanent impacts. Unlike
the LQR framework or the classical Almgren-Chriss model, where dynamics are assumed to follow
some stochastic model, here we run the policy gradient method directly on the LOB without any
assumption on how the mid-price St moves and what are the forms of the market impacts. Denote
by qt = qt−1 −ut−1 the inventory at time t. We restrict the admissible controls to be of the linear
feedback form ut = −Kt(St, qt)⊤with some Kt ∈R1×2.
The cost ct = φ′(qt −ut)2 −rt(ut) at time t consists of two parts. The ﬁrst part φ′(qt −ut)2 is the
holding cost of the inventory weighted by a parameter φ′. The quantity rt(ut) is the amount we receive
by liquidating ut shares at time t. Note that rt(·) may depend on St and other market observables.
For example, if we liquidate ut = 1000 shares of the asset with the market conditions given in Table 1,
then the amount received would be
rt(ut) = 397 × 200.1 + 412 × 200.0 + (1000 −397 −412) × 199.9 = 200020.6.
This transaction moves the best bid price two levels down.
This is commonly referred to as the
temporary impact of a market order.
Bid level
One
Two
Three
Four
Five
Bid price (USD)
200.1
200.0
199.9
199.8
199.7
Volume available
Table 1: One snapshot of the LOB.
Performance Metric: Implementation Shortfall [40]
IS(u) =
T−1
X
t=0
ct(ut) + cT

q0 −
T−1
X
t=0
ut
!!
−c0(q0).
(5.2)
The ﬁrst term of (5.2) is the cost of implementing policy u over the horizon [0, T]. The second term
is the cost when liquidating q0 market orders at time 0. If we expect u is better than liquidating
everything at time 0, then IS(u) < 0. A smaller implementation shortfall implies the strategy is more
proﬁtable.
We use the following relative performance (evaluated on a single trajectory) to compare the perfor-
mance of two policies u1 and u2,
Relative performance = IS(u2) −IS(u1)
|IS(u2)|
.
Experiment Set-up.
We consider the LOB data consisting of the best 5 levels and we assume the
trading frequency ∆= 1 minute and the trading horizon T = 10 minutes. We perform a numerical
analysis for ﬁve diﬀerent stocks, Apple (AAPL), Facebook (FB), International Business Machines Cor-
poration (IBM), American Airlines (AAL) and JP Morgan (JPM), during the period from 01/01/2019
to 12/31/2019. The data is divided into two sets, a training set with data between 10:00AM-12:00AM
01/01/2019-08/31/2019 and a test set with data between 10:00AM-12:00AM 09/01/2019-12/31/2019.
We take φ′ = 5 × 10−6; T = 10; smoothing parameter r = 0.4; number of trajectories m = 200;
initial policy K 0 ∈R1×20 with (K 0)ij = −0.2 for all i, j; and step size η = 10−6. We assume the initial
inventory follows q0 = 2000. We compare the performance of the policy gradient method with the
Almgren-Chriss solution with ﬁtted parameters given in Table 3. In the Almgren-Chriss model, we set
φ = σ2φ′ to ensure a reasonable comparison.

Results.
From Table 2 and Figure 5, the policy gradient method improves on the Almgren-Chriss
solution by around 20% on ﬁve diﬀerent stocks from diﬀerent ﬁnancial sectors. Note that the goal
of the policy gradient method is to learn the global minimizer of the expected cost function, hence it
is expected that the Almgren-Chriss solution could perform better than the policy gradient method
for some sample trajectories, as shown in Figure 5. This result is compatible with the performance
of the Q-learning algorithms [30]. The drawback of Q-learning algorithms is that the computational
complexity is highly dependent on the size of the set of (discrete) states and actions, where as the
policy gradient method can handle continuous states and actions.
We conjecture that the policy gradient method may be capable of learning the global “optimal”
solution for a larger class of models that are “similar” to the LQR framework with stochastic dynamics
and ﬁnite time horizon. In addition, as the policy gradient method is a model-free algorithm, it is more
robust with respect to model mis-speciﬁcation as compared to the Almgren-Chriss framework.
(a) IBM.
(b) AAL.
(c) JPM.
(d) FB.
(e) AAPL.
Figure 5: Empirical distribution of the relative performance on the test set.
Asset
IBM
AAL
JPM
FB
AAPL
In sample
0.173
0.152
0.251
0.181
0.165
(std)
(0.09)
(0.27)
(0.31)
(0.32)
(0.31)
Out of sample
0.178
0.146
0.245
0.175
0.163
(std)
(0.08)
(0.29)
(0.36)
(0.24)
(0.37)
Table 2: Average relative performance of the policy gradient (u1) compared to Almgren-Chriss solution
(u2).
5.3
Learning LQR in Higher Dimensions
In practice we can perform the policy gradient method for the optimal liquidation problem with multiple
assets. However it is diﬃcult to capture the cross impact and permanent impact with historical LOB
data.
Therefore we test the performance of the policy gradient method in higher dimensions on

synthetic data consisting of a four-dimensional state variable and a two-dimensional control variable.
The parameters are randomly picked such that the conditions for our LQR framework are satisﬁed.
Set-up.
(1) Parameters:
A =




0.5
0.05
0.1
0.2
0.2
0.3
0.1
0.06
0.1
0.2
0.4
0.05
0.2
0.15
0.1



, B =




−0.05
−0.01
−0.005
−0.01
−1
−0.01
−0.01
−0.9



, Qt =




0.5
−0.01
−0.1
1.1
0.2
0.1
0.9
−0.06
0.03
−0.1
0.88



,
Rt =
 0.4
−0.2
−0.3
0.7

, W =




0.1
0.5
0.2
0.3



,
QT = Qt, T = 10; smoothing parameter r = 1, number of trajectories m = 200; initial policy
K 0 ∈R2×40 with {K 0}ij = 0.05 for all i, j, for both known and unknown parameters;
(2) Initialization: We assume x0 = (x1
0, x2
0, x3
0, x4
0)⊤and xi
0 are independent. x1
0, x2
0, x3
0, and x4
0 are
sampled from N(5, 0.1), N(2, 0.3), N(8, 1), N(5, 0.5).
Convergence.
For the high-dimensional case, the normalized error falls below the threshold 10−2
within 80 iterations for the policy gradient algorithm with known parameters. It takes substantially
more iterations for the policy gradient algorithm with unknown parameters to have an error near such
a threshold, which is as expected.
(a) Known parameters
(η = 0.0005).
(b) Unknown parameters
(η = 0.0001).
Figure 6: Performance of the policy gradient algorithms
(50 simulation scenarios)
Figure 7: Performance of the pol-
icy gradient algorithm with un-
known parameters under diﬀer-
ent step size η (50 simulation sce-
narios).
Outcomes from Varying the Parameter η.
The performance of the policy gradient algorithm also
depends on the values of the step size η. We show how the values of the step size η ∈[10−5, 2 × 10−3]
aﬀect the convergence of the policy gradient algorithm with unknown parameters in Figure 7. A tiny
step size leads to slow convergence (see the blue line when η = 10−5) and a larger step size may cause
divergence (see the purple line when η = 2 × 10−3).

## References
[1] Yasin Abbasi-Yadkori and Csaba Szepesvári. Regret bounds for the adaptive control of linear
quadratic systems. In Proceedings of the 24th Annual Conference on Learning Theory, pages 1–26,
2011.
[2] Marc Abeille and Alessandro Lazaric. Thompson sampling for linear-quadratic control problems.
AISTATS 2017 - 20th International Conference on Artiﬁcial Intelligence and Statistics, 2017.
[3] Marc Abeille, Alessandro Lazaric, Xavier Brokmann, et al. LQG for portfolio optimization. Avail-
able at SSRN 2863925, 2016.
[4] Radoslaw Adamczak. A note on the Hanson-Wright inequality for random vectors with depen-
dencies. Electronic Communications in Probability, 20, 2015.
[5] Aurélien Alfonsi, Antje Fruth, and Alexander Schied. Optimal execution strategies in limit order
books with general shape functions. Quantitative Finance, 10(2):143–157, 2010.
[6] Alessandro Alla and Valeria Simoncini. Order reduction approaches for the algebraic riccati equa-
tion and the lqr problem. In Numerical Methods for Optimal Control Problems, pages 89–109.
Springer, 2018.
[7] Robert Almgren. Optimal execution with nonlinear impact functions and trading-enhanced risk.
Applied Mathematical Finance, 10(1):1–18, 2003.
[8] Robert Almgren and Neil Chriss. Optimal execution of portfolio transactions. Journal of Risk,
3:5–40, 2001.
[9] Robert Almgren, Chee Thum, Emmanuel Hauptmann, and Hong Li. Direct estimation of equity
market impact. Risk, 18(7):58–62, 2005.
[10] Brian D. O. Anderson and John B Moore. Optimal Control: Linear Quadratic Methods. Courier
Corporation, 2007.
[11] Karl J Åström and Björn Wittenmark. Adaptive control. Courier Corporation, 2013.
[12] Wenhang Bao and Xiao-yang Liu. Multi-agent deep reinforcement learning for liquidation strategy
analysis. arXiv preprint arXiv:1906.11046, 2019.
[13] Dimitri Bertsekas. Dynamic Programming And Optimal Control, volume 1. Athena Scientiﬁc, 3rd
edition, 2005.
[14] Jalaj Bhandari and Daniel Russo. Global optimality guarantees for policy gradient methods. arXiv
preprint arXiv:1906.01786, 2019.
[15] Jingjing Bu, Afshin Mesbahi, Maryam Fazel, and Mehran Mesbahi. LQR through the lens of ﬁrst
order methods: discrete-time case. arXiv preprint arXiv:1907.08921, 2019.
[16] Jingjing Bu, Afshin Mesbahi, and Mehran Mesbahi.
Policy gradient-based algorithms for
continuous-time linear quadratic control. arXiv preprint arXiv:2006.09178, 2020.
[17] Jingjing Bu, Lillian J Ratliﬀ, and Mehran Mesbahi. Global convergence of policy gradient for
sequential zero-sum linear quadratic dynamic games. arXiv preprint arXiv:1911.04672, 2019.

[18] René Carmona, Mathieu Laurière, and Zongjun Tan. Linear-quadratic mean-ﬁeld reinforcement
learning: convergence of policy gradient methods. arXiv preprint arXiv:1910.04295, 2019.
[19] Arthur Charpentier, Romuald Elie, and Carl Remlinger. Reinforcement learning in economics and
ﬁnance. arXiv preprint arXiv:2003.10014, 2020.
[20] Rama Cont, Arseniy Kukanov, and Sasha Stoikov. The price impact of order book events. Journal
of Financial Econometrics, 12(1):47–88, 2014.
[21] Sarah Dean, Horia Mania, Nikolai Matni, Benjamin Recht, and Stephen Tu.
On the sample
complexity of the linear quadratic regulator. Foundations of Computational Mathematics, pages
1–47, 2019.
[22] Mohamad Kazem Shirani Faradonbeh, Ambuj Tewari, and George Michailidis. Optimism-based
adaptive regulation of linear-quadratic systems. IEEE Transactions on Automatic Control, 2020.
[23] Salar Fattahi, Nikolai Matni, and Somayeh Sojoudi.
Eﬃcient learning of distributed linear-
quadratic control policies. SIAM Journal on Control and Optimization, 58(5):2927–2951, 2020.
[24] Maryam Fazel, Rong Ge, Sham M Kakade, and Mehran Mesbahi. Global convergence of pol-
icy gradient methods for the linear quadratic regulator.
Proceedings of the 35th International
Conference on Machine Learning, pages 1467–1476, 2018.
[25] Claude-Nicolas Fiechter. PAC adaptive control of linear systems. In Proceedings of the Tenth
Annual Conference on Computational Learning Theory, pages 72–80, 1997.
[26] Jim Gatheral and Alexander Schied. Optimal trade execution under geometric Brownian motion
in the Almgren and Chriss framework. International Journal of Theoretical and Applied Finance,
14(03):353–368, 2011.
[27] Benjamin Gravell, Peyman Mohajerin Esfahani, and Tyler Summers.
Learning robust con-
trollers for linear quadratic systems with multiplicative noise via policy gradient. arXiv preprint
arXiv:1905.13547, 2019.
[28] David Gross. Recovering low-rank matrices from few coeﬃcients in any basis. IEEE Transactions
on Information Theory, 57(3):1548–1566, 2011.
[29] Xin Guo, Renyuan Xu, and Thaleia Zariphopoulou. Entropy regularization for mean ﬁeld games
with learning. arXiv preprint arXiv:2010.00145, 2020.
[30] Dieter Hendricks and Diane Wilcox. A reinforcement learning extension to the Almgren-Chriss
framework for optimal trade execution. In 2014 IEEE Conference on Computational Intelligence
for Financial Engineering & Economics (CIFEr), pages 457–464. IEEE, 2014.
[31] Morteza Ibrahimi, Adel Javanmard, and Benjamin V Roy. Eﬃcient reinforcement learning for
high dimensional linear quadratic systems. In Advances in Neural Information Processing Systems,
pages 2636–2644, 2012.
[32] Zeyu Jin, Johann Michael Schmitt, and Zaiwen Wen. On the analysis of model-free methods for
the linear quadratic regulator. arXiv preprint arXiv:2007.03861, 2020.
[33] Laura Leal, Mathieu Laurière, and Charles-Albert Lehalle.
Learning a functional control for
high-frequency ﬁnance. arXiv preprint arXiv:2006.09611, 2020.

[34] Weiwei Li and Emanuel Todorov. Iterative linear quadratic regulator design for nonlinear biological
movement systems. In ICINCO, pages 222–229, 2004.
[35] Dhruv Malik, Ashwin Pananjady, Kush Bhatia, Koulik Khamaru, Peter Bartlett, and Martin
Wainwright.
Derivative-free methods for policy optimization: guarantees for linear quadratic
systems. In The 22nd International Conference on Artiﬁcial Intelligence and Statistics, pages
2916–2925. PMLR, 2019.
[36] Yuriy Nevmyvaka, Yi Feng, and Michael Kearns.
Reinforcement learning for optimized trade
execution. In Proceedings of the 23rd International Conference on Machine Learning, pages 673–
680, 2006.
[37] Brian Ning, Franco Ho Ting Ling, and Sebastian Jaimungal. Double deep Q-learning for optimal
execution. arXiv preprint arXiv:1812.06600, 2018.
[38] Yi Ouyang, Mukul Gagrani, and Rahul Jain. Control of unknown linear systems with Thompson
sampling. In 2017 55th Annual Allerton Conference on Communication, Control, and Computing
(Allerton), pages 1198–1205. IEEE, 2017.
[39] Panagiotis Patrinos, Sergio Trimboli, and Alberto Bemporad.
Stochastic MPC for real-time
market-based optimal power dispatch. In 2011 50th IEEE Conference on Decision and Control
and European Control Conference, pages 7111–7116. IEEE, 2011.
[40] Andre F Perold. The implementation shortfall: Paper versus reality. Journal of Portfolio Man-
agement, 14(3):4, 1988.
[41] Andreas Schmidt and Bernard Haasdonk. Reduced basis approximation of large scale parametric
algebraic riccati equations. ESAIM: Control, Optimisation and Calculus of Variations, 24(1):129–
151, 2018.
[42] Stephen Tu and Benjamin Recht.
Least-squares temporal diﬀerence learning for the linear
quadratic regulator. In International Conference on Machine Learning, pages 5005–5014, 2018.
[43] Yasuaki Wasa, Kengo Sakata, Kenji Hirata, and Kenko Uchida.
Diﬀerential game-based load
frequency control for power networks and its integration with electricity market mechanisms. In
2017 IEEE Conference on Control Technology and Applications (CCTA), pages 1044–1049. IEEE,
2017.
[44] Zhuoran Yang, Yongxin Chen, Mingyi Hong, and Zhaoran Wang.
On the global conver-
gence of actor-critic: a case for linear quadratic regulator with ergodic cost.
arXiv preprint
arXiv:1907.06246, 2019.
[45] Kaiqing Zhang, Zhuoran Yang, and Tamer Basar. Policy optimization provably converges to Nash
equilibria in zero-sum linear quadratic games.
In Advances in Neural Information Processing
Systems, pages 11602–11614, 2019.
[46] Zihao Zhang, Stefan Zohren, and Stephen Roberts. Deep reinforcement learning for trading. The
Journal of Financial Data Science, 2(2):25–40, 2020.

A
Market Simulator for Linear Price Dynamics
We estimate the parameters for the LQR model using NASDAQ ITCH data taken from Lobster1.
Permanent Price Impact and Volatility
The model in (2.7) implies that prices changes are
proportional to the market-order ﬂow imbalances (MFI). We adopt the framework from [20], namely
that the price change ∆S is given by
∆S = γ MFI + σ ϵ,
(A.1)
with MFI = Mb −Ms where Ms and Mb are the volumes of market sell orders and market buy orders
respectively during a time interval ∆T = 5mins and ϵ ∼N(0, 1). We then estimate γ and σ from the
data.
Figure 8: Relationship between MFI and ∆S. (Example (from left to right): AAP, FB, JPM, IBM
and AAL, 10:00AM-11:00AM 01/01/2019-08/31/2019, ∆T = 1min)
Temporary Price Impact
We assume the LOB has a ﬂat shape with constant queue length l for the
ﬁrst few levels. Figure 9 shows the average queue lengths for the ﬁrst 5 levels so that our assumption
is not too unreasonable. Therefore the following equation, on the amount received when we liquidate
u shares with best bid price S, holds
u(S −βu) =
Z S
S−u ∆
l
lvdv.
Therefore we have β = ∆
2l, where ∆is the tick size and l is the average queue length.
Figure 9: Average queue length (volume) of the ﬁrst ﬁve levels on the limit buy side (Example (from
left to right): AAP, FB, JPM, IBM and AAL, 10:00AM-11:00AM 01/01/2019-08/31/2019 with 5000
samples uniformly sampled with natural time clock in each trading day.)
Parameter Estimation
See the estimates for AAPL, FB, IBM, JPM, and AAL in Table 3.
1https://lobsterdata.com/

Paramters/Stock
AAPL
FB
IBM
JPM
AAL
β
1.03 × 10−5
1.30 × 10−5
2.65 × 10 ∗∗−5
9.28 × 10−6
3.27 × 10−5
γ
7.27 × 10−6
1.40 × 10−5
4.60 × 10−5
1.65 × 10−5
1.3310 × 10−5
σ
0.107
0.115
0.082
0.059
0.042
Table 3:
Parameter estimation from NASDAQ ITCH Data (10:00AM-11:00PM 01/01/2019-
08/31/2019).
B
Proofs of Technical Results
We now give the proofs that were omitted in the text.
B.1
Proofs in Section 3.1
Proof of Lemma 3.2. Denote by {xt}T
t=0 the state trajectory induced by an arbitrary control K. By
Assumption 3.1 the matrix E[x0x⊤
0 ] is positive deﬁnite. For t ≥1, we have
E[xtx⊤
t ] = (A −BKt−1)E[xt−1x⊤
t−1](A −BKt−1)⊤+ E[wt−1w⊤
t−1].
Now (A−BKt−1)E[xt−1x⊤
t−1](A−BKt−1)⊤is positive semi-deﬁnite and E[wt−1w⊤
t−1] is positive deﬁnite.
Hence E[xtx⊤
t ] is positive deﬁnite and as a result σX > 0. In this case, we can simply take σX =
min(E[x0x⊤
0 ], σmin(W)).
Proof of Proposition 3.4. This can be proved by backward induction. For t = T, PK
T
= QT is
positive deﬁnite since QT is positive deﬁnite. Assume PK
t+1 is positive deﬁnite for some t+1, then take
any z ∈Rd such that z ̸= 0,
z⊤PK
t z = z⊤Qt z + z⊤K⊤
t RtKtz + z⊤(A −BKt)⊤PK
t+1 (A −BKt) z > 0.
The last inequality holds since z⊤Qt z > 0, z⊤K⊤
t RtKtz ≥0 and z⊤(A −BKt)⊤PK
t+1 (A −BKt) z ≥
0. By backward induction, we have PK
t
positive deﬁnite, ∀t = 0, 1, · · · , T.
To prove Lemma 3.6, let us start with a useful result for the value function. Deﬁne the value
function VK(x, τ) for τ = 0, 1, · · · , T −1, as
VK(x, τ) = Ew
"T−1
X
t=τ
(x⊤
t Qtxt + u⊤
t Rtut) + x⊤
T QT xT

xτ = x
#
= x⊤Pτx + Lτ,
with terminal condition
VK(x, T) = x⊤QT x,
where Lτ is deﬁned in (3.10). We then deﬁne the Q function, QK(x, u, τ) for τ = 0, 1, · · · , T −1 as
QK(x, u, τ) = x⊤Qτx + u⊤Rτu + Ewτ [VK(Ax + Bu + wτ, τ + 1)] ,
and the advantage function
AK(x, u, τ) = QK(x, u, τ) −VK(x, τ).
Note that C(K) = Ex0∼D[V (x0, 0)]. Then we can write the diﬀerence of value functions between K
and K ′ in terms of advantage functions.

Lemma B.1. Assume K and K ′ have ﬁnite costs. Denote {x′
t}T
t=0 and {u′
t}T−1
t=0 as the state and
control sequences of a single trajectory generated by K ′ starting from x′
0 = x0 = x, then
VK ′(x, 0) −VK(x, 0) = Ew
"T−1
X
t=0
AK(x′
t, u′
t, t)
#
,
(B.1)
and
AK(x, −K′
τx, τ)
= 2x⊤(K′
τ −Kτ)⊤Eτx + x⊤(K′
τ −Kτ)⊤(Rτ + B⊤Pτ+1B)(K′
τ −Kτ)x,
where Eτ is deﬁned in (3.11).
Proof. Denote by c′
t(x) the cost generated by K ′ with a single trajectory starting from x′
0 = x0 = x.
That is,
c′
t(x) = (x′
t)⊤Qtx′
t + (u′
t)⊤Rtu′
t, t = 0, 1, · · · , T −1,
and
c′
T (x) = (x′
T )⊤QT x′
T ,
with
u′
t = −K′
tx′
t,
x′
t+1 = Ax′
t + Bu′
t + wt,
x′
0 = x.
Therefore,
VK ′(x, 0) −VK(x, 0) = Ew
" T
X
t=0
c′
t(x)
#
−VK(x, 0)
= Ew
" T
X
t=0
 c′
t(x) + VK(x′
t, t) −VK(x′
t, t)

#
−VK(x, 0)
= Ew
"T−1
X
t=0
 c′
t(x) + VK(x′
t+1, t + 1) −VK(x′
t, t)

#
= Ew
"T−1
X
t=0
 QK(x′
t, u′
t, t) −VK(x′
t, t)


x0 = x
#
= Ew
"T−1
X
t=0
AK(x′
t, u′
t, t)

x0 = x
#
,
where the third equality holds since c′
T (x) = VK(x′
T , T) with the same single trajectory. For u = −K′
τx,

AK(x, −K′
τx, τ) = QK(x, −K′
τx, τ) −VK(x, τ)
= x⊤(Qτ + (K′
τ)⊤RτK′
τ)x + Ewτ

VK((A −BK′
τ)x + wτ, τ + 1)

−VK(x, τ)
= x⊤(Qτ + (K′
τ)⊤RτK′
τ)x +

x⊤(A −BK′
τ)⊤Pτ+1(A −BK′
τ)x + Tr(WPτ+1) + Lτ+1

−

x⊤Pτx + Lτ

= x⊤(Qτ + (K′
τ −Kτ + Kτ)⊤Rτ(K′
τ −Kτ + Kτ))x
+ x⊤(A −BKτ −B(K′
τ −Kτ))⊤Pτ+1(A −BKτ −B(K′
τ −Kτ))x
−x⊤(Qτ + K⊤
τ RτKτ + (A −BKτ)⊤Pτ+1(A −BKτ))x
= 2x⊤(K′
τ −Kτ)⊤((Rτ + B⊤Pτ+1B)Kτ −B⊤Pτ+1A)x
+ x⊤(K′
τ −Kτ)⊤(Rτ + B⊤Pτ+1B)(K′
τ −Kτ)x.
(B.2)
Proof of Lemma 3.6. First for any K′
τ, from (B.2),
AK(x, −K′
τx, τ) = QK(x, −K′
τx, τ) −VK(x, τ)
= 2 Tr(xx⊤(K′
τ −Kτ)⊤Eτ) + Tr(xx⊤(K′
τ −Kτ)⊤(Rτ + B⊤Pτ+1B)(K′
τ −Kτ))
= Tr
 xx⊤(K′
τ −Kτ + (Rτ + B⊤Pτ+1B)−1Eτ)⊤(Rτ + B⊤Pτ+1B)
(K′
τ −Kτ + (Rτ + B⊤Pτ+1B)−1Eτ)

−Tr(xx⊤E⊤
τ (Rτ + B⊤Pτ+1B)−1Eτ)
≥−Tr(xx⊤E⊤
τ (Rτ + B⊤Pτ+1B)−1Eτ),
(B.3)
with equality holds when K′
τ = Kτ −(Rτ + B⊤Pτ+1B)−1Eτ. Then,
C(K) −C(K ∗)
=
−E
T−1
X
t=0
AK(x∗
t , u∗
t , t)
≤
E
T−1
X
t=0
Tr

x∗
t (x∗
t )⊤E⊤
t (Rt + B⊤Pt+1B)−1Et

≤
∥ΣK ∗∥
T−1
X
t=0
Tr(E⊤
t (Rt + B⊤Pt+1B)−1Et)
≤
∥ΣK ∗∥
σR
T−1
X
t=0
Tr(E⊤
t Et)
(B.4)
≤
∥ΣK ∗∥
4 σX 2 σR
T−1
X
t=0
Tr(∇tC(K)⊤∇tC(K)),
(B.5)
where σX is deﬁned in (3.3) and σR is deﬁned in (3.4). For the lower bound, consider K′
t = Kt −(Rt +

B⊤Pt+1B)−1Et where the equality holds in (B.3). Using C(K ∗) ≤C(K ′)
C(K) −C(K ∗) ≥C(K) −C(K′
K′
K′)
= −E
T−1
X
t=0
AK(x′
t, u′
t, t)
= E
T−1
X
t=0
Tr(x′
t(x′
t)⊤E⊤
t (Rt + B⊤Pt+1B)−1Et)
≥σX
T−1
X
t=0
∥Rt + B⊤Pt+1B∥Tr(E⊤
t Et)
(B.6)
Proof of Lemma 3.7. By lemma B.1 we have
C(K ′) −C(K) = E
"T−1
X
t=0
AK(x′
t, −K′
tx′
t, t)
#
=
T−1
X
t=0

2 Tr(Σ′
t(K′
t −Kt)⊤Et) + Tr(Σ′
t(K′
t −Kt)⊤(Rt + B⊤Pt+1B)(K′
t −Kt))

.
Proof of Lemma 3.8. For t = 0, 1, · · · , T,
C(K) ≥E[x⊤
t Ptxt] ≥∥Pt∥σmin(E[xtx⊤
t ]) ≥σX ∥Pt∥,
C(K) =
T−1
X
t=0
Tr(E[xtx⊤
t ](Qt + K⊤
t RtKt)) + Tr(E[xT x⊤
T ]QT ) ≥σQ Tr(ΣK) ≥σQ ∥ΣK∥.
Therefore the statement in Lemma 3.8 follows provided that σX > 0 and Assumption 2.1 holds.
Proof of Proposition 3.9 . Recall that Σt = E

xtx⊤
t

. Note that
Σ1
=
E
h
x1x⊤
i
= E
h
((A −B K0)x0 + w0) ((A −B K0)x0 + w0)⊤i
=
(A −B K0)Σ0 (A −B K0)⊤+ W = G0(Σ0) + W.
Now we ﬁrst prove that
Σt = Gt−1(Σ0) +
t−1
X
s=1
Dt−1,sWD⊤
t−1,s + W, ∀t = 2, 3, · · · , T.
(B.7)
When t = 2,
Σ2
=
E
h
x2x⊤
i
= E
h
((A −B K1)x1 + w1) ((A −B K1)x1 + w1)⊤i
=
(A −B K1)Σ1 (A −B K1)⊤+ W = G1(Σ0) + (A −BK1)W(A −BK1)⊤+ W,

which satisﬁes (B.7). Assume (B.7) holds for t ≤k. Then for t = k + 1,
E
h
xt+1x⊤
t+1
i
=
E
h
((A −B Kt)xt + wt) ((A −B Kt)xt + wt)⊤i
=
(A −B Kt)Σt (A −B Kt)⊤+ W = Gt(Σ0) +
t
X
s=1
Dt,sWD⊤
t,s + W.
Therefore (B.7) holds, ∀t = 1, 2, · · · , T. Finally,
ΣK =
T
X
t=0
Σt = Σ0 +
T−1
X
t=1
Gt(Σ0) +
T−1
X
t=1
t
X
s=1
Dt,sWD⊤
t,s + TW = TK(Σ0) + ∆(K, W).
B.2
Proofs in Section 3.2
Proof of Lemma 3.13. By direct calculation,
∥Gt∥≤ρ2(t+1),
and
∥G′
t∥≤ρ2(t+1).
(B.8)
Denote Ft = FKt and F′
t = FK′
t to ease the exposition. Then for any symmetric matrix Σ ∈Rd×d and
t ≥0,
∥(G′
t+1 −Gt+1)(Σ)∥
=
∥F′
t+1 ◦G′
t(Σ) −Ft+1 ◦Gt(Σ)∥
=
∥F′
t+1 ◦G′
t(Σ) −F′
t+1 ◦Gt(Σ) + F′
t+1 ◦Gt(Σ) −Ft+1 ◦Gt(Σ)∥
≤
∥F′
t+1 ◦G′
t(Σ) −F′
t+1 ◦Gt(Σ)∥+ ∥F′
t+1 ◦Gt(Σ) −Ft+1 ◦Gt(Σ)∥
=
∥F′
t+1 ◦(G′
t −Gt)(Σ)∥+ ∥(F′
t+1 −Ft+1) ◦Gt(Σ)∥
≤
∥F′
t+1∥∥(G′
t −Gt)(Σ)∥+ ∥Gt∥∥F′
t+1 −Ft+1∥∥Σ∥
≤
ρ2∥(G′
t −Gt)(Σ)∥+ ρ2(t+1)∥F′
t+1 −Ft+1∥∥Σ∥.
Therefore,
∥(G′
t+1 −Gt+1)(Σ)∥≤ρ2∥(G′
t −Gt)(Σ)∥+ ρ2(t+1)∥F′
t+1 −Ft+1∥∥Σ∥.
(B.9)
Summing (B.9) up for t ∈{1, 2, · · · , T −2} with ∥G′
0 −G0∥= ∥F′
0 −F0∥, we have
T−1
X
t=0

(Gt −G′
t)(Σ)

≤ρ2T −1
ρ2 −1
 T−1
X
t=0
∥Ft −F′
t∥

∥Σ∥.
B.3
Proofs in Section 3.3
Proof of Lemma 3.15. Given (3.22) and condition (3.23), we have
∥K′
t −Kt∥= η∥∇tC(K)∥≤
σQ σX
2C(K)∥B∥.
Therefore,
∥B∥∥K′
t −Kt∥≤σQ σX
2C(K) ≤1
2.

The last inequality holds since σX ≤C(K)
σQ
given by Lemma 3.8. Therefore, by Lemma 3.12,
T−1
X
t=0
∥FKt −FK′
t∥≤(2ρ + 1)∥B∥
T−1
X
t=0
∥Kt −K′
t∥
!
.
(B.10)
By Lemmas 3.5 and 3.7,
C(K ′) −C(K) =
T−1
X
t=0
h
2 Tr

Σ′
t(K′
t −Kt)⊤Et

+ Tr

Σ′
t(K′
t −Kt)⊤(Rt + B⊤Pt+1B)(K′
t −Kt)
i
=
T−1
X
t=0
h
−4η Tr

Σ′
tΣtE⊤
t Et

+ 4η2 Tr

Σ′
tΣtE⊤
t (Rt + B⊤Pt+1B)EtΣt
i
=
T−1
X
t=0
h
−4η Tr

(Σ′
t −Σt + Σt)ΣtE⊤
t Et

+ 4η2 Tr

Σ′
tΣtE⊤
t (Rt + B⊤Pt+1B)EtΣt
i
≤
T−1
X
t=0
h
−4η Tr

ΣtE⊤
t EtΣt

+ 4η Tr((Σ′
t −Σt)ΣtE⊤
t EtΣtΣ−1
t )
+ 4η2 Tr

Σ′
tΣtE⊤
t (Rt + B⊤Pt+1B)EtΣt
i
≤
T−1
X
t=0
h
−4η Tr

ΣtE⊤
t EtΣt

+ 4η∥Σ′
t −Σt∥
σmin(Σt) Tr

ΣtE⊤
t EtΣt

+ 4η2∥Σ′
t(Rt + B⊤Pt+1B)∥Tr

ΣtE⊤
t EtΣt
i
≤−η

1 −
PT−1
t=0 ∥Σ′
t −Σt∥
σX
−η∥ΣK ′∥
T−1
X
t=0
∥Rt + B⊤Pt+1B∥
 T−1
X
t=0
h
Tr(∇tC(K)⊤∇tC(K))
i
.
(B.11)
By Lemma 3.6, we have
C(K ′) −C(K) ≤−η

1 −
PT−1
t=0 ∥Σ′
t −Σt∥
σX
−η∥ΣK ′∥
T−1
X
t=0
∥Rt + B⊤Pt+1B∥
4 σX
2 σR
∥ΣK ∗∥

C(K) −C(K ∗)

(B.12)
provided that
1 −
PT−1
t=0 ∥Σ′
t −Σt∥
σX
−η∥ΣK ′∥
T−1
X
t=0
∥Rt + B⊤Pt+1B∥> 0.
(B.13)
By (3.21), (3.22), and (B.10),
T−1
X
t=0
∥Σ′
t −Σt∥≤ρ2T −1
ρ2 −1
C(K)
σQ
+ T∥W∥

η(2ρ + 1)∥B∥
T−1
X
t=0
∥∇tC(K)∥
!
.
Given the step size condition in (3.23), we have
η(2ρ+1)∥B∥
T−1
X
t=0
∥∇tC(K)∥≤η(2ρ+1)∥B∥

T ·max
t {∥∇tC(K)∥}

≤
(ρ2 −1) σQ σX
2(ρ2T −1)(C(K) + σQ T∥W∥).
(B.14)

Then, by Corollary 3.14 and (B.10),
∥ΣK ′ −ΣK∥
σX
≤
ρ2T −1
ρ2 −1
 T−1
X
t=0
∥FKt −FK′
t∥
∥Σ0∥+ T∥W∥
σX
≤
ρ2T −1
ρ2 −1 (2ρ + 1)∥B∥
T−1
X
t=0
η∥∇tC(K∥
!
C(K) + σQ T∥W∥
σQ σX
≤
2,
where the last step holds by (B.14). Therefore, the bound of ∥ΣK ′∥in (B.13) is given by
∥ΣK ′∥≤∥ΣK ′ −ΣK∥+ ∥ΣK∥≤1
2 σX +C(K)
σQ
≤1
2∥ΣK ′∥+ C(K)
σQ
,
(B.15)
which indicates that ∥ΣK ′∥≤2C(K)
σQ
. Therefore, (B.13) gives
1 −
PT−1
t=0 ∥Σ′
t −Σt∥
σX
−η∥ΣK ′∥
T−1
X
t=0
∥Rt + B⊤Pt+1B∥
≥1 −
(ρ2T −1)
(ρ2 −1) σX
C(K)
σQ
+ T∥W∥

η(2ρ + 1)∥B∥
T−1
X
t=0
∥∇tC(K)∥
!
−η2C(K)
σQ
T−1
X
t=0
∥Rt + B⊤Pt+1B∥
= 1 −C1η,
where C1 is deﬁned in (3.24). So if η ≤
2C1 , then,
1 −
PT−1
t=0 ∥Σ′
t −Σt∥
σX
−η∥ΣK ′∥
T−1
X
t=0
∥Rt + B⊤Pt+1B∥≥1 −C1η ≥1
2 > 0.
Hence,
C(K ′) −C(K) ≤−η
4 σX
2 σR
∥ΣK ∗∥

C(K) −C(K ∗)

,
and
C(K ′) −C(K ∗) =
 C(K ′) −C(K)

+ (C(K) −C(K ∗)) ≤

1 −2ησX
2 σR
∥ΣK ∗∥

C(K) −C(K ∗)

.
B.4
Proofs in Section 4
Proof of Lemma 4.6.
Under Assumption 4.3,
E
h
x0x⊤
i
= f
W0E
h
z0z⊤
i
f
W ⊤
0 ,

E
h
x0x⊤
i

≤σ2
0∥f
W0∥2.
With the sub-Gaussian distributed noise,
W = E
h
wtw⊤
t
i
= f
WE
h
vtv⊤
t
i
f
W ⊤,

then we have ∥W∥≤σ2
w

f
W 2

.
Denote St = Qt + KT
t RtKt, ∀t = 1, · · · , T −1. Thus, for t = 0, 1, · · · , T −2,
E[x⊤
t+1Qt+1xt+1 + u⊤
t+1Rt+1ut+1] = E[x⊤
t+1St+1xt+1] = Tr(E[x⊤
t+1St+1xt+1]) = Tr(E[xt+1x⊤
t+1]St+1)
= Tr

Gt(Σ0)St+1 +
t
X
s=1
Dt,sWD⊤
t,sSt+1 + WSt+1
!
.
The last equality holds by (B.7). Therefore,
C(K ′) −C(K) = E[x⊤
0 (K′
0)⊤R0K′
0x0 −x⊤
0 K⊤
0 R0K0x0]
|
{z
}
(I)
+
T−2
X
t=0
Tr

G′
t(Σ0)S′
t+1 −Gt(Σ0)St+1

|
{z
}
(II)
+
T−2
X
t=0
Tr

t
X
s=1

D′
t,sW(D′
t,s)⊤S′
t+1 −Dt,sWD⊤
t,sSt+1

+ W(S′
t+1 −St+1)

|
{z
}
(III)
+ Tr

GT−1(Σ0)QT −G′
T−1(Σ0)QT +
T−1
X
s=1

D′
T−1,sW(D′
T−1,s)⊤QT −DT−1,sWD⊤
T−1,sQT
!
|
{z
}
(IV )
.
For the ﬁrst term (I),
(I) ≤Tr(E[x0x⊤
0 ])∥(K′
0)⊤R0K′
0 −K⊤
0 R0K0∥
For the second term (II), since
T−2
X
t=0
(Tr (Gt(Σ0)St+1)) = E
"T−2
X
t=0

Tr

Πt
i=0(A −BKi)x0x⊤
0 Πt
i=0(A −BKt−i)⊤St+1
#
≤Tr

E
h
x0x⊤
i

T−2
X
t=0
Gt(St+1)

,
we have,
(II) ≤Tr

E
h
x0x⊤
i

T−2
X
t=0
 G′
t
 S′
t+1

−Gt (St+1)


.

We denote Gd := PT−2
t=0
 G′
t
 S′
t+1

−Gt (St+1)

, then
∥Gd∥≤
T−2
X
t=0

G′
t

Qt+1 + (K′
t+1)⊤Rt+1K′
t+1

−Gt

Qt+1 + (K′
t+1)⊤Rt+1K′
t+1

−
Gt ◦

K⊤
t+1Rt+1Kt+1 −(K′
t+1)⊤Rt+1K′
t+1


≤ρ2T −1
ρ2 −1

(2ρ + 1)∥B∥
T−2
X
t=0
∥Kt −K′
t∥
! T−1
X
t=1
∥Qt + (K′
t)⊤RtK′
t∥
!
+
T−2
X
t=0
∥Gt∥

(K′
t+1)⊤Rt+1K′
t+1 −K⊤
t+1Rt+1Kt+1

≤ρ2T −1
ρ2 −1

(2ρ + 1)∥B∥
T−2
X
t=0
∥Kt −K′
t∥
! T−1
X
t=1
∥Qt + (K′
t)⊤RtK′
t −K⊤
t RtKt + K⊤
t RtKt∥
!
+ ρ2(ρ2(T−1) −1)
ρ2 −1
T−1
X
t=1

(K′
t)⊤RtK′
t −K⊤
t RtKt

≤ρ2T −1
ρ2 −1 (2ρ + 1)∥B∥

K ′ −K


|||Q||| + |||K|||2 |||R|||

+

ρ2T −1
ρ2 −1 (2ρ + 1)∥B∥

K ′ −K

+ ρ2(ρ2(T−1) −1)
ρ2 −1
! T−1
X
t=1

(K′
t)⊤RtK′
t −K⊤
t RtKt

.
(B.16)
where the second inequality holds by Lemma 3.13 and (B.10), and the third inequality holds by (B.8).
For the ﬁrst term in (III), we have
T−2
X
t=0
Tr
t
X
s=1
D′
t,sW(D′
t,s)⊤S′
t+1 −Dt,sWD⊤
t,sSt+1
!
=
T−2
X
t=0
Tr
t
X
s=1
D′
t,sW(D′
t,s)⊤(S′
t+1 −St+1) + (D′
t,sW(D′
t,s)⊤−Dt,sWD⊤
t,s)St+1
!
≤
 T−2
X
t=0
t
X
s=1
Tr(W)∥D′
t,s∥2

T−1
X
t=1
(K′
t)⊤RtK′
t −K⊤
t RtKt

+
T−2
X
t=0

t
X
s=1
D′
t,sW(D′
t,s)⊤−Dt,sWD⊤
t,s

 T−1
X
t=1
Tr(Qt) + ∥Kt∥2 Tr(Rt)

≤
Tr(W)(T −1)(ρ2(T−1) −1)
ρ2 −1

T−1
X
t=1
(K′
t)⊤RtK′
t −K⊤
t RtKt

+T (ρ2T −1)
ρ2 −1 (2ρ + 1)∥B∥∥W∥

K ′ −K

Tr
T−1
X
t=1
Qt
!
+ |||K|||2 Tr
T−1
X
t=1
Rt
!!
,
where the last step holds by (3.20). The second term in (III) is bounded by
T−2
X
t=0
Tr

W(S′
t+1 −St+1)

≤Tr(W)
T−1
X
t=1

(K′
t)⊤RtK′
t −K⊤
t RtKt

.

Similarly, by (3.20) and (B.10), (IV ) is bounded by
(IV )
≤
Tr(E[x0x⊤
0 ])
T−1
X
t=0

(G′
t −Gt)(QT )

+ Tr
T−1
X
s=1
D′
T−1,sW(D′
T−1,s)⊤QT −DT−1,sWD⊤
T−1,sQT
!
≤
Tr(E[x0x⊤
0 ])ρ2T −1
ρ2 −1 (2ρ + 1)∥B∥∥QT ∥

K ′ −K

+ Tr(QT )ρ2T −1
ρ2 −1 (2ρ + 1)∥B∥∥W∥

K ′ −K

.
Now we bound the term PT−1
t=1

(K′
t)⊤RtK′
t −K⊤
t RtKt

, which appears several times in previous
inequalities:
T−1
X
t=1

(K′
t)⊤RtK′
t −K⊤
t RtKt

=
T−1
X
t=1

(K′
t −Kt + Kt)⊤Rt(K′
t −Kt + Kt) −K⊤
t RtKt

≤
T−1
X
t=1
∥K′
t −Kt∥2∥Rt∥+ 2∥Kt∥∥Rt∥∥K′
t −Kt∥
≤3|||K||| |||R|||

K ′ −K

.
The last step holds since ∥K′
t −Kt∥≤∥Kt∥by assumption.
Therefore,
|C(K ′) −C(K)| ≤Tr(E[x0x⊤
0 ])
n
3|||K|||∥R0∥

K ′ −K

+ ρ2T −1
ρ2 −1 (2ρ + 1)∥B∥∥QT ∥

K ′ −K

+ ρ2T −1
ρ2 −1 (2ρ + 1)∥B∥

K ′ −K


|||Q||| + |||K|||2 |||R|||

+

ρ2T −1
ρ2 −1 (2ρ + 1)∥B∥

K ′ −K

+ ρ2(1 −ρ2(T−1))
ρ2 −1
!
3|||K||| |||R|||

K ′ −K

o
+ 3Tr(W)
(T −1)(ρ2(T−1) −1)
ρ2 −1
+ 1

|||K||| |||R|||

K ′ −K

+

T (ρ2T −1)
ρ2 −1 (2ρ + 1)∥B∥∥W∥

K ′ −K


Tr
T−1
X
t=1
Qt
!
+ |||K|||2 Tr
T−1
X
t=1
Rt
!!
+ Tr(QT )ρ2T −1
ρ2 −1 (2ρ + 1)∥B∥∥W∥

K ′ −K

.
By (3.27), Lemma 3.8, and Lemma 3.16, ρ is bounded above by polynomials in ∥A∥, ∥B∥, |||R|||,
σX ,
σR and C(K), or a constant 1 + ξ. Therefore, we rewrite the above inequality by
|C(K ′) −C(K)| ≤hCK

K ′ −K

+ h′
CK

K ′ −K

2,
(B.17)
where hCK ∈H(C(K)) and h′
CK ∈H(C(K)) are polynomials in C(K) and model parameters. Given
assumption (4.5), we have |||K ′ −K||| ≤1 and hence

K ′ −K

≥

K ′ −K

2.
Deﬁne hcost = hCK + h′
CK, then (B.17) gives
|C(K ′) −C(K)| ≤hcost

K ′ −K

,
with hcost ∈H(C(K)).

Proof of Lemma 4.7. Recall ∇tC(K) = 2EtΣt and W = E

wtw⊤
t

= f
WE

vtv⊤
t
 f
W ⊤. We have,
∥∇tC(K ′) −∇tC(K)∥= ∥2E′
tΣ′
t −2EtΣt∥≤2∥E′
t −Et∥∥Σ′
t∥+ 2∥Et∥∥Σ′
t −Σt∥,
(B.18)
For the second term, by Lemma 3.6 and Cauchy-Schwarz inequality,
∥Et∥≤
T−1
X
t=0
∥Et∥≤
T−1
X
t=0
q
Tr(E⊤
t Et) ≤
s
T · maxt ∥Rt + B⊤Pt+1B∥
σX
(C(K) −C(K ∗)).
(B.19)
By (B.9) and direct calculation, we have
∥(G′
t+1 −Gt+1)(Σ0)∥≤ρ2(t+1)
t+1
X
s=0
∥FK′s −FKs∥∥Σ0∥
!
.
By (B.10) and (3.20), for t = 1, 2, · · · , T −1,
∥Σ′
t −Σt∥≤∥(G′
t −Gt)(Σ0)∥+

t−1
X
s=0
Dt−1,sWD⊤
t−1,s −D′
t−1,sW(D′
t−1,s)⊤

≤ρ2t(2ρ + 1)∥B∥∥Σ0∥

K ′ −K

+ (ρ2T −1)
ρ2 −1 (2ρ + 1)∥B∥∥W∥

K ′ −K

.
(B.20)
Therefore the second term in (B.18) is bounded by the product of (B.19) and (B.20).
Next we bound the ﬁrst term in (B.18). Similar to (B.15), ∥Σ′
t∥≤∥PT
t=0 Σ′
t∥= ∥ΣK ′∥≤∥Σ′
K −
ΣK∥+ ∥ΣK∥≤
C(K)
σQ
+ ∥ΣK∥.
For ∥E′
t −Et∥, we ﬁrst need a bound on ∥P ′
t −Pt∥.
Since P0 =
S0 + PT−2
t=0 Gt(St+1) + GT−1(QT ), by (B.16), we have
∥P ′
t −Pt∥≤∥P ′
0 −P0∥≤3∥K0∥∥R0∥∥K′
0 −K0∥+ ∥Gd∥+ ρ2T −1
ρ2 −1 (2ρ + 1)∥B∥∥QT ∥
T−1
X
t=0
∥Kt −K′
t∥
!
≤ρ2T −1
ρ2 −1 (2ρ + 1)∥B∥

K ′ −K


|||Q||| + |||K|||2|||R|||

+ 3

1 + ρ2T −1
ρ2 −1 (2ρ + 1)∥B∥

K ′ −K

+ ρ2(1 −ρ2(T−1))
ρ2 −1
!
· |||K||| |||R|||

K ′ −K

+ ρ2T −1
ρ2 −1 (2ρ + 1)∥B∥∥QT ∥

K ′ −K

.
(B.21)
Thus,

E′
t −Et

=

Rt(K′
t −Kt) −B⊤(P ′
t+1 −Pt+1)A + B⊤(P ′
t+1 −Pt+1)BK′
t + B⊤Pt+1B(K′
t −Kt)

≤
 ∥Rt∥+ ∥B∥2∥P0∥


K ′ −K

+ ∥B∥∥P ′
0 −P0∥∥A∥+ 2∥B∥2∥P ′
0 −P0∥|||K|||.
Given the bound on |||K||| = PT−1
t=0 ∥Kt∥in Lemma 3.16 and the bound on ∥Pt∥in Lemma 3.8, all the
terms in (B.18) can be bounded by polynomials of related parameters multiplied by |||K ′ −K||| and
|||K ′ −K|||2. Similarly to the proof of Lemma 4.6, we have |||K ′ −K||| ≤1 and
∥∇tC(K ′) −∇tC(K)∥≤hgrad

K ′ −K

,
for some polynomial hgrad ∈H(C(K)).

B.5
Proofs in Section 5
Proof of Proposition 5.2. Denote Ht :=
1 + γk1
t
γk2
t
k1
t
1 + k2
t

. Since Ht has two eigenvalues 1 and γk1
t +
k2
t + 1, Ht is positive deﬁnite when γk1
t + k2
t > −1 (0 ≤t ≤T −1).
Then let us show the ﬁrst claim by induction. Assume E[xsx⊤
s ] is positive deﬁnite for all s ≤t,
then
E[xt+1x⊤
t+1]
=
E[((A −BKt)xt + wt) ((A −BKt)xt + wt)⊤]
=
E[(Htxt + wt) (Htxt + wt)⊤]
=
E[Htxtx⊤
t H⊤
t + wtw⊤
t + wtw⊤
t + 2Htxtw⊤
t ]
=
HtE[xtx⊤
t ]H⊤
t +
σ

.
Hence E[xt+1x⊤
t+1] is positive deﬁnite since E[xtx⊤
t ] is positive deﬁnite and Ht is positive deﬁnite.
Therefore σX > 0.
The second claim can be proved by backward induction. For t = T, PK
T = QT is positive deﬁnite
since QT is positive deﬁnite. Assume PK
t+1 is positive deﬁnite for some t + 1, then take any z ∈Rd
such that z ̸= 0,
z⊤P K
t z = z⊤Qt z + z⊤K⊤
t RtKtz + z⊤H⊤
t PK
t+1Htz > 0.
Note that Ht is positive deﬁnite when γk1
t + k2
t > −1 and 1 + γk1
t > 0. The last inequality holds
since Qt and H⊤
t PK
t+1Ht are positive deﬁnite, and K⊤
t RtKt is positive semi-deﬁnite. Hence we have
PK
t
positive deﬁnite for all t = 0, 1, 2, · · · , T.
