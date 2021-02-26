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


# Introduction

## Overview

1. The deep exploration problem
2. Our ambition: Rapid RL
2. Approaches
3. Abstraction + novelty


## Deep Exploration

Sparse rewards, too much experience required by most (esp. deep) solutions.


## Our ambition: Rapid RL

### Direct challenge
> Learn Daniel Willemsen's environment in three trajectories.

### Stretch goal
> Learn any simple exploration problem quickly.


## Approaches

![Figure title](images/cwi-logo.png)

### Related approaches
- Novelty search
- Curiosity
- Optimism & value function randomisation

## Abstraction + novelty

Why is abstraction helping

What form of novelty do we estimate

# Method




## Contribution

Combining estimators
