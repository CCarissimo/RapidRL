## Introduction

The goal of this project is to explore techniques that can be used to speed up reinforcement learning, seeking for useful behavior above optimal behavior. We call this Rapid Reinforcement Learning. To this end we try a complementary combination of exploratory methods, and multi-level abstractions to improve performance in simulated games.

## Content

Main contents of the repository are:
- agents: folder containing scripts necessary to construct the reinforcement learning agents we test on our environments
- environments: folder containing scripts to generate environments on which we test our agents
- results: folder where we store results pertaining to different experiments
- utils: folder that stores some useful scripts for running, analyzing and plotting
- main.py: python scripts used to run experiments that sits in the root directory
- experiments.md: markdown file where we store the commands to run experiments to show the effects and improvements of our methods
- rapidrl_overview.md: markdown file where we describe the equations and methods that we apply in our experiments

Link to the overleaf https://www.overleaf.com/project/5fcbc0e6c47c2fc0475ba5b9

The main.py script in the root folder is used to run all the experiments. Parameters for the run are to be specified on the command line. You can access instructions to use the command line tool by running the help function of main.py:
> python main.py test DEATH SIMPLE_NOVELTOR --plotGW -n 10000 --timeout 10000 --noexploit --gamma 0.2 --plot --animate
