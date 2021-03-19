---
title: "Rapid reinforcement learning"
subtitle: "Deep exploration by novelty abstraction"
author: \textbf{Cesare Carissimo and Michael Kaisers} -- Intelligent and Autonomous Systems research group Centrum Wiskunde & Informatica
date: "5 March 2021"
theme: "metropolis"
titlegraphic: images/cwi-logo.png
beameroption: ""
themeoptions: "numbering=fraction,progressbar=frametitle,sectionpage=none"
fontsize: 11pt
output:
  beamer_presentation:
    slide_level: 1
  pdf_document:
      toc: false
export_on_save:
  pandoc: true
header-includes:
  - \usepackage{bm}
  - \usepackage{bbm}
  - \usepackage[normalem]{ulem}
  - \usepackage{mathpazo}\usepackage{ragged2e}
  - \usepackage{framed}\usepackage{xcolor}\colorlet{shadecolor}{black!10}
  - \renewenvironment{quote}{\begin{shaded*}\justifying\footnotesize}{\end{shaded*}}
---


<!--
# Abstract
Deep reinforcement learning is great with abundant resources and many samples. Reality however confronts agents with limited resources, sparse rewards and deceptive feedback. We need rapid agents that explore effectively, by finding sparse rewards, and efficiently, by using few samples. Recent work has shown that novelty as a proxy for "interestingness" can be used to guide exploration in an effective manner. Novelty search biases exploration towards behaviors that are potentially interesting. Additionally, abstraction techniques can help make efficient use of samples. Abstraction is successful when it constrains an agent to focus on features that are most influential by abstracting out irrelevant information. Our contribution towards effective and efficient deep exploration is novelty guided exploration with abstraction.
-->
<!--![Figure title](images/cwi-logo.png) -->


# Outline

1. Our Ambition: Rapid Reinforcement Learning
2. Why Deep Exploration is Hard
    - Sparsity
    - Deception
    - Bounded Resources
3. Approaches
    - Abstraction
    - Novelty & Variants
4. Contributions
    - Abstraction over Novelty
    - Combination of Abstractions
5. Discussion


# Our Ambition: Rapid Reinforcement Learning

![This exploration problem is surprisingly hard for vanilla RL.](images/willemsen_grid.png)

## Our ambitions
1. Solve this in three trajectories,
2. Explore with emergent policies of gradually increasing complexity, and finally
3. Sample efficient discovery of sparse rewards.


# Deep Exploration

Finding rewards that are:

 1. Sparse

 2. Deceptive

with Bounded Resources.

## The Problem(s) with Objectives

Lehman, Stanley (2011)


# Deep Exploration - Sparsity

![This is the deep sea exploration environment. Because agents only learns when rewarded, infrequent rewards lead to infrequent learning.](images/deep_sea.png){ width=75% }


# Deep Exploration - Deception

> Sometimes a mutation increases fitness but actually leads further from the objective. (Goldberg, 1987)

![A simple environment where following the objective gradient may lead to a deceptive reward.](images/deceptive_reward.png)


# Deep Exploration - Bounded Resources

> We can not practically rely on convergence in the limit.


# Approach - Abstraction

![Abstraction generalises at the cost of bias. Abstractions levels range from the identity (single cell) via local (3x3) to global contexts. Abstract estimates may be combined based on their uncertainty and bias.](images/abstractors.png)

<!-- Contexts over transition histories are also possible. -->

# Approach - Abstraction

An Abstractor $\mathcal{A}: (C, f, V, N)$

$C$ set of contexts;

$f: \mathcal{H} \times \mathcal{A} \xrightarrow{} C$, function mapping histories to contexts;

$V: \mathcal{C} \times \mathcal{A} \xrightarrow{} \mathbb{R}$, function mapping action values in context;

$N: \mathcal{C} \times \mathcal{A} \xrightarrow{} \mathbb{N}$, function counting samples used to estimate the abstraction value, a proxy for uncertainty.

Baier, Kaisers (2021)


# Approach - Multi Level Abstractions

$V^C(c, a) = V^C(f(x, a))$, for all C in $\{identity, column, 3\times 3, global\}$

![For a single state x we can estimate value over multiple abstractions. We can then choose to combine them to produce a single value estimate for state x.](images/abstractors_combined.png)


# Approach - Auxiliary-Objective Search

DO NOT: pursue reward maximizing behavior

DO: pursue 'novel' behavior

via intrinsic rewards


# Approach - Intrinsic Rewards

![Recall our discussion of Munchausen RL. A reward is extrinsic when it comes from the environment. A reward is intrinsic when it comes from the agent. An intrinsic reward can be mixed with extrinsic rewards as an Optimistic Bonus term.](images/intrinsic_reward.png)


# Approach - Auxiliary-Objective Search

> Criticism: internal reward just another objective

> Rebuttal: novelty rewards are based on past behaviors and mostly orthogonal to external rewards

Lehman, Stanley (2011)

Best when always changing, non-stationary, diverging.


# Approach - Count Based Exploration

SUBJECT: $\{f_i\}_{i \in M, t}:$ features

MEASURE: $\rho_t = \sum_{i \in M} P(f_i)$

Count occurrences of features and construct a density model for the learned features in an environment.

Martin et. al. (2017)


# Approach - Novelty

> New is not always Interesting, but Interesting is always New.  

SUBJECT: $x:$ behavior

MEASURE: $\rho(x) = \frac{1}{k} \sum_{i=0}^k dist(x,\mu_i)$

The average distance of a behavior to its k-nearest behaviors.

Lehman, Stanley (2011)

Video: [Novelty-Biped](https://www.youtube.com/watch?v=dXQPL9GooyI&t=120s)


# Approach - Diversity Driven RL

SUBJECT: $\pi:$ policy

MEASURE: $L_D = L - E_{\pi'\in\Pi} [\alpha D(\pi, \pi')]$

The difference of a policy to other policies stored in memory.

$D$ could be KL divergence, MSE, L2-norm ...

Hong et. al. (2018)


# Approach - Curiosity

SUBJECT: $\pi:$ policy

MEASURE: $||\pi(x_t, a_t) - \varpi(x_{t+1})||_2^2$

Mean squared error against a fixed variance gaussian density.

Burda et. al. (2018)


# Approach - Approximating Novelty

When behavior space is intractable, continuous, multidimensional: LARGE

SUBJECT: $x:$ behavior passed through an auto-encoder

MEASURE: $\rho(x) = \frac{1}{k} \sum_{i=0}^k dist(x,\mu_i)$

Ramamurthy et. al. (2020)


# Approach - Curiosity Bottleneck

> Noisy-TV Problem: Pure randomness is always novel, and truly un-interesting

SUBJECT: $p_{\theta}(Z|X):$ compressor model

FIND: $\min_{\theta} -I(Z;Y) + \beta I(X;Z)$

MEASURE: $\rho(x) = KL [p_{\theta}(Z|x) || q(Z)]$

Kim et. al. (2019)


# Contributions - Abstraction over Novelty

![Where $Q(s,a)$ is an estimation of discounted future rewards, $\mathcal{N}(s,a)$ is an estimation of discounted future novelty, using a count-based technique for estimating uncertainty about an $(s,a)$ pair. We can compute an estimate of the Novelty of a state over the multi level abstractors we hold for each context.](images/abstractors_novelty.png)


# Contributions - Combining Abstractions

![Here we show the tradeoff between variance and bias for abstractors with contexts of increasing size. We can use variance and bias to weigh larger contexts heavier when agents have observed fewer transitions.](images/novelty_linearly_combined.png)


# Method - Estimation

We estimate and store discounted future expectations for Reward and Novelty values. Both ground estimates can be abstracted over. 

Both estimates are updated with a bellman like equation:

$Q(s,a) = Q(s,a) + \alpha(r + \gamma (V(s')))$, where $Q(s,a)$ is the action-value function $V(s)$ is the state-value function.

All abstractions of the same estimate use the same targets to update.  


# Method - Estimation

The bellman update is optimal if the values we would like to estimate have optimal substructure, that is we assume that we can write the value of our decision problem as a recursive specification over the value induced by our current action from a given state and the value of the next state. 

$V(x_o) = \max_{\{a_t\}_{t=0}^{\inf}}\sum_{t=0}^{\inf}\beta^t F(x_t, a_t)$, where $F(x_t, a_t)$ is the reward 

NB: While environment rewards are fixed, novelty rewards change in relation to experience; returning to a state x that had novelty $n_t$ at will be less novel, $n_{t+\tau} < n_t$. Therefore, for novelty the optimal state-value function is non-stationary. 


# Method - Estimation 

$N: \mathcal{S} \times \mathcal{A} \xrightarrow{} \mathbb{N}$, (s,a) pair observation counter

'Knownness' is approximated as $K(s,a) = \frac{1}{N(s,a)}$. We then use knownness as the learning rate for estimating expected future rewards. We further estimate future learning by using K as a reward for our second estimator. 
  
## Expected Sum of Discounted Future Rewards 

$Q(s,a) = Q(s,a) + K(s,a) [r + \gamma(\max_{a'}Q(s',a') - Q(s,a))]$

## Expected Sum of Discounted Sum of Future Learning 

$\mathcal{N}(s,a) = \mathcal{N}(s,a) + \alpha [K(s,a) + \gamma(\max_{a'}\mathcal{N}(s',a') - \mathcal{N}(s,a))]$


# Method - Action Selection 

Our action selection design must follow some main principles:

1. start: 
    - with simple policies
    - by prioritising exploration
    
2. as experience increases:
    - increase policy complexity
    - shift priority towards exploitation

When confronted with the unknown we want our agent to envision a hypothesis, test it and update its beliefs. On the other hand, when confronted with the 'known' we want our agent to consider acting greedily.


# Method - Action Selection 

![To achieve our goal of solving this environment in a few trajectories we expect only the global and identity abstractors to be sufficient. So in the next slide we will only consider how to balance the (s,a) estimate and the global estimate of novelty in exploration.](images/abstractor_global_identity.png)


# Method - Action Selection 

As first step we simplify our problem as a two phase process, and will only consider Novelty estimates. 

Phase 1. Explore using novelty. If in an unseen state use the global abstractor. If in a seen state apply a \textbf{non-greedy} and \textbf{optimistic} strategy on the action-value estimates.

$\pi^1(s) = \mathbbm{1}{\{\max_{a}N(s,a) > 0\}} arg\max_{a}\mathcal{N}(s,a) + \mathbbm{1}{\{\max_{a}N(s,a) = 0\}} arg\max_{a}\mathcal{N}^{global}(a)$

Phase 2. Exploit using novelty. Pick the maximum action-value estimate in each state.  

$\pi^2(s) = arg\max_{a}\mathcal{N}(s,a)$


# Theory - Combining Rewards

An alternative approach to a phased action selection is to combine rewards. We have an extrinsic reward $r^e$ given by the environment rewards, and an intrinsic reward $r^i$ given by knownness scores. First off we must decide how to combine these rewards. We can combine rewards into one as $r = r^e + r^i$, possibly including weights for each term. We can imagine creating a much more complicated reward that combines several intrinsic rewards: 

$$r = r^e + r^{\text{identity}} + r^{\text{column}} + \dots + r^{\text{global}}$$

Alternatively we can hold separate estimators for each reward. In the case of abstractions over rewards we may only need to estimate the action value function which we can then use to estimate our abstractions. What I would like to show next is that these two approaches are distinct, and that even if we combine our estimates when we select our actions in the same way as we combine the rewards in the other method we do not have equivalent processes. 


# Theory - Combining State-Value Functions

Consider the value function $V_{\pi}(s) = E_{\pi}[G_t|S_t=s]$ where $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$ is the discounted sum of future rewards. 

If we combine our rewards $r^c = r^e + r^i$, we get the following expansion for our value function:

$$V_{\pi}^c(s) = \sum_{a}\pi(a|s)\sum_{s',r^e,r^i}p(s',r^e,r^i|s,a)[r^e + r^i + \gamma V_{\pi}(s')]$$


# Theory - Combining State-Value Functions

On the other hand, if we keep two separate estimates for intrinsic and extrinsic rewards:

$$V_{\pi}^e(s) = \sum_{a}\pi(a|s)\sum_{s',r^e}p(s',r^e|s,a)[r^e + \gamma V_{\pi}^e(s')]$$
$$V_{\pi}^i(s) = \sum_{a}\pi(a|s)\sum_{s',r^i}p(s',r^i|s,a)[r^i + \gamma V_{\pi}^i(s')]$$

we can then ask, does $V_{\pi}^c = V_{\pi}^e + V_{\pi}^i$ ?

<!--$$V_{\pi}^e + V_{\pi}^i = \sum_{a}\pi(a|s)\sum_{s',r^e,r^e}p(s',r^e,r^i|s,a)[r^e + r^i + \gamma (V_{\pi}^e(s') + V_{\pi}^i(s'))]$$-->


# Theory - Combining State-Value Functions

Given, $R_{t+1} = R^e_{t+1} + R^i_{t+1}$, then:

$$V_{\pi}^c(s) = E[G_t|S_t = s] = E[\sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}|S_t = s]$$

$$= E[\sum_{k=0}^{T-t-1} \gamma^k R^e_{t+k+1} + R^i_{t+k+1}|S_t = s]$$

$$= E[\sum_{k=0}^{T-t-1} \gamma^k R^e_{t+k+1} + \sum_{k=0}^{T-t-1} \gamma^k R^i_{t+k+1}|S_t = s]$$


# Theory - Combining State-Value Functions

by linearity of expectation,

$$= E[\sum_{k=0}^{T-t-1} \gamma^k R^e_{t+k+1}|S_t = s] + E[\sum_{k=0}^{T-t-1} \gamma^k R^i_{t+k+1}|S_t = s]$$

$$= (V_{\pi}^e(s) + V_{\pi}^i(s))$$

So there we have it. If we linearly combine the estimators it does not matter whether we combine the rewards or the estimates. What is of course important that we are using the same policy, which is the case because we end up combining the estimates after we have updated them separately. 

If we were to do any kind of non-linear $f$ combination for which we can not factorize $f(r_e, r_i)$ this equivalence does not hold. 


# Theory - Combining State-Value Functions

I reiterate here that the reason why this equivalence holds for value functions is that we are using the same policy in both cases, where the policy itself is over the combination of estimators. What I mean is that the update of the state value function for extrinsic and intrinsic rewards is done using the same policy for the one-step lookahead target as we do when updating the single state value function. 

We can see how we get intro trouble with this when we do not have the value function, and must estimate it using action-value functions. 


# Theory - Combining Q Functions

Consider $Q_{\pi}^c(s,a) = \sum_{s',r^e,r^i}p(s',r^e,r^i|s,a)[r^e + r^i + \gamma V_{\pi}(s')]$ as the action-value function for the combined reward $r=r^e + r^i$. Since we do not know the state-value function $V_{\pi}$ we, assume we will update with respect to the greedy policy $\pi(a) = arg \max_{a} Q_{\pi}(s,a)$, and rewrite as:

$$ Q_{\pi}^c(s,a) = \sum_{s',r^e,r^i}p(s',r^e,r^i|s,a)[r^e + r^i + \gamma \max_{a'} Q_{\pi}(s',a')] $$
 

# Theory - Combining Q Functions

All math works out very similarly as for state-value functions when we consider holding two separate estimates for extrinsic and intrinsic rewards and combining then after the update, but for one big difference:

$$Q^c_{\pi}(s,a) = Q^e_{\pi}(s,a) + Q^i_{\pi}(s,a)$$
$= \sum_{s',r^e}p(s',r^e|s,a)[r^e + \gamma \max_{a'} Q^e_{\pi}(s',a')] + \sum_{s',r^i}p(s',r^i|s,a)[r^i + \gamma \max_{a'} Q^i_{\pi}(s',a')]$
$$= \sum_{s',r^e,r^i}p(s',r^e,r^i|s,a)[r^e + r^i + \gamma (\max_{a'} Q^e_{\pi}(s',a') + \max_{a'} Q^i_{\pi}(s',a'))]$$


# Theory - Combining Q Functions

Recall the last line of the previous slide: 
$$\sum_{s',r^e,r^i}p(s',r^e,r^i|s,a)[r^e + r^i + \gamma (\max_{a'} Q^e_{\pi}(s',a') + \max_{a'} Q^i_{\pi}(s',a'))]$$

which is only equal to $Q_{\pi}(s,a)$ if 

$$\max_{a}[Q_{\pi}(s',a')] = \max_{a'} Q^e_{\pi}(s',a') + \max_{a'} Q^i_{\pi}(s',a')$$ 
$$\max_{a}[Q^e_{\pi}(s',a') + Q^i_{\pi}(s',a')] = \max_{a'} Q^e_{\pi}(s',a') + \max_{a'} Q^i_{\pi}(s',a')$$ 


# Theory - Combining Q Functions 
Generally:
$$\max_{a}[Q^e_{\pi}(s',a') + Q^i_{\pi}(s',a')] <= \max_{a'} Q^e_{\pi}(s',a') + \max_{a'} Q^i_{\pi}(s',a')$$ 

with equivalence only if we pick an action a for our separate intrinsic and extrinsic updates that maximises both metrics simultaneously. 

Maintaining two separate estimates while running policy improvements can lead us to a situation where the greedy policy with resect to the different rewards results in two different actions $a_e \neq a_i$. We may thus overestimate the value of a state with respect to the greedy policy of the combined rewards. This also results in off-policy learning, since the polices we use to update may differ from the ones we use to act. 


# Theory - Linear Combination of $V_{\pi}$

Consider a more general case of combination where we combine our rewards with a weight vector $\lambda$. 

$$V^c_{\pi}(s) = \lambda^T \vec{V} = \sum_{i \in \lambda} \lambda_i V^i_{\pi}(s)$$
$$= \sum_{i \in \lambda} \lambda_i E_{\pi}[G_t|S_t=s] =  \sum_{i \in \lambda} E_{\pi}[\lambda_i G_t|S_t=s]$$
$=  E_{\pi}[\sum_{i \in \lambda} \lambda_i G_t|S_t=s] = V_{\pi}(s)$$

when $R_t = \lambda^T \vec(R)$.


# Discussion

1. How else can we combine Abstractions?
2. Can we deal with a changing environment?
3. How does this fit in with Open Ended Learning?
