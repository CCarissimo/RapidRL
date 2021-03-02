---
title: "Deep Exploration"
subtitle: "Challenges and solutions of reinforcement learning"
author: \textbf{Cesare Carissimo and Michael Kaisers} -- Intelligent and Autonomous Systems research group Centrum Wiskunde & Informatica
date: "tbd March 2021"
theme: "metropolis"
titlegraphic: images/cwi-logo.png
beameroption: ""
themeoptions: "numbering=fraction,progressbar=frametitle,sectionpage=none"
fontsize: 11pt
output:
  beamer_presentation:
    slide_level: 2
  pdf_document:
      toc: false
export_on_save:
  pandoc: true
header-includes:
  - \usepackage{bm}
  - \usepackage[normalem]{ulem}
  - \usepackage{mathpazo}\usepackage{ragged2e}
  - \usepackage{framed}\usepackage{xcolor}\colorlet{shadecolor}{black!10}
  - \renewenvironment{quote}{\begin{shaded*}\justifying\footnotesize}{\end{shaded*}}
---


# Abstract
Deep reinforcement learning is great with abundant resources and many samples. Reality however confronts agents with limited resources, sparse rewards and deceptive feedback. We need rapid agents that explore effectively, by finding sparse rewards, and efficiently, by using few samples. Recent work has shown that novelty as a proxy for "interestingness" can be used to guide exploration in an effective manner. Novelty search biases exploration towards behaviors that are potentially interesting. Additionally, abstraction techniques can help make efficient use of samples. Abstraction is successful when it constrains an agent to focus on features that are most influential by abstracting out irrelevant information. Our contribution towards effective and efficient deep exploration is novelty guided exploration with abstraction.

![Figure title](images/cwi-logo.png)


# Introduction


## Overview

1. The Challenge of Deep Exploration
2. Approaches
3. Abstractions on Novelty


## Deep Exploration

Finding Sparse Rewards,
with Bounded Resources
in Deceptive Environments.


# Our ambition: Rapid RL


## Direct challenge
> Learn Daniel Willemsen's environment in three trajectories.

add willemsen gridworld image

## Stretch goal
> Learn any simple exploration problem quickly.


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


# Extension - Novelty 

When behavior space is intractable, continuous, multidimensional: LARGE

SUBJECT: $x:$ behavior passed through an auto-encoder

MEASURE: $\rho(x) = \frac{1}{k} \sum_{i=0}^k dist(x,\mu_i)$

Ramamurthy et. al. (2020)


# Extension - Curiosity Bottleneck 

> Noisy-TV Problem: Pure randomness is always novel, and truly un-interesting

SUBJECT: $p_{\theta}(Z|X):$ compressor model

FIND: $\min_{\theta} -I(Z;Y) + \beta I(X;Z)$

MEASURE: $\rho(x) = KL [p_{\theta}(Z|x) || q(Z)]$

Kim et. al. (2019)


# Approach - Value Function Randomization



# Method




## Contribution

Combining estimators
