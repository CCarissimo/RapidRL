# Rapid RL

author: Cesare Carissimo

### How do we know we are done? 

#### Past

The original goal for our project was to solve the Daniel Willemsen gridworld in three trajectories. The follow up goal was to have a decision making strategy that could work for generalized sparse reward settings. A side goal was for the decision making strategy to increase in complexity as it increased experience. 

#### Present

As of now we have several implementations of tabular q learning that tackle these goals. We are currently combining multiple estimators with the akaike information criterion. These estimators differ in abstraction levels and in parameter counts. We use parameter counts to estimate complexity of the estimator, and the residual sum of squares to estimate the accuracy of the estimator. Each estimator has a single RSS value for all contexts it considers. We are now making sure that all the behavior is as expected so that we can say something meaningful about the system we are implementing. 

##### Estimators

- Q-table
  - identity
  - row
  - column
  - global
- N-table
  - identity
  - row
  - column
  - global
- Linear
  - reward
  - novelty
- Combined MSE Estimator
- Combined AIC Estimator

##### Recent bug-fixes:

- We minimize AIC = 2k + n ln(RSS), and n is now set to the total updates for an estimator, not state specific, since we only have one RSS value for each estimator
- prev_W in predict is now only saved when we select actions. We were predicting on all states to save data in the training loop and this was messing up our RSS values for weights. 

#### Future

We would like to show that our estimator combination works better for decision making under uncertainty, and thus works better for exploration. We would like to implement an exploration scheme that is not based on randomness, but rather is based on abstraction and generalisation. We need to define what we see as exploiting and what we see as exploring. We may create another gridworld that we can use to show the effectiveness of our estimator combinations. We will be done if our estimator combinations are able to successfully solve an environment many times faster and more efficiently than a vanilla q table learner. Solving an environment we can define as getting the maximum payoff when exploiting. 

##### A) Performance and complexity of Combined Estimation under $\epsilon$-greedy exploration

###### Illustration of the reward plateau

$AIC_i = 2k_i + n \ln(RSS_i)$, where k is the number of fitted parameters (including 1 global variance parameter)

$W_i = \exp((AIC_{min}-AIC_{i})/2)$

Potential Extensions:

- Dense Reward Environments
- State specific variance/RSS estimates
- Dueling Q-learning (with value estimates) as baseline
- SARSA
- Expected SARSA
- Boltzmann Exploration

Baselines:

- Dueling Q-Learning

Experiments:

- STRAIGHT gridworld with Identity, Global, Linear estimators

- WILLEMSEN gridworld with Identity, Global, Linear estimators
- POOL gridworld with Identity, Global, Row, Column, Linear

##### B) Add Novelty as an Intrinsic Reward

$N = 1/n_s$

Extensions:

- $N = \exp(-n_s)$
- $N = n_s^{-b}$
- Pseudo-count novelty

##### C) Effect of Novelty based Exploration with multi-level Novelty Estimates

Using Novelty specific Estimators to define the exploration policy, then exploiting with reward based estimators. 

##### Novelty

Novelty is currently implemented as 'knowness' measuring the reciprocal of state action visits. We would like to expand this to pseudo count novelties so that we can cite relevant literatures for it to justify the inclusion. I would use novelty as follows:

- Exploration with Novelty
- Exploitation with Rewards

so our done conditions are satisfied when following rewards leads us to the maximum payoff. We hope that our novelty based exploration strategy will allow our agent to find the states that have high rewards so that it can learn how to exploit best. We may leave for future reserach how we should balance novelty based exploration and reward based exploitation online. I would ideally like to devise such a process.

##### Estimators 

We would like to add to our list of estimators:

- Quadratic estimation
- negative exponential estimation

in the same way that we implemented a linear estimator.

