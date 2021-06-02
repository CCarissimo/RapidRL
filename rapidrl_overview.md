# Rapid RL

author: Cesare Carissimo

### How do we know we are done? 

#### Past

The original goal for our project was to solve the Daniel Willemsen gridworld in three trajectories. The follow up goal was to have a decision making strategy that could work for generalized sparse reward settings. A side goal was for the decision making strategy to increase in complexity as it increased experience. 

#### Present

As of now we have several implementations of tabular q learning that tackle these goals. We are currently combining multiple estimators with the akaike information criterion. These estimators differ in abstraction levels and in parameter counts. We use parameter counts to estimate complexity of the estimator, and the residual sum of squares to estimate the accuracy of the estimator. Each estimator has a single RSS value for all contexts it considers. 

