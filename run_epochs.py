#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Agents import *
from Environments import *
from Experiments import *
from Utils import *
from warnings import filterwarnings
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import pickle as cPickle


# In[2]:


EXPERIMENT = 'A'
GRIDWORLD = 'WILLEMSEN'
AGENT_TYPE = 'ME_Q'
PLOT = True
ANIMATE = False
MAX_STEPS = 10000
EPISODE_TIMEOUT = 33
GAMMA = 0.9
ALPHA = 0.1
RSS_ALPHA = 0.1
LIN_ALPHA = 0.0001
BATCH_SIZE = 10
WEIGHTS_METHOD = 'exp_size_corrected'
EXPLOIT = True
EPSILON = 0.1
EPOCHS = 100  # 0.159 for 100 EPOCHS should be 1.59 for 1000 EPOCHS. 
dim2D = False if GRIDWORLD == "WILLEMSEN" else True
cwd = os.getcwd()
FOLDER = "%s\\Results" % (cwd)
FILE_SIG = f"{EXPERIMENT}_{AGENT_TYPE}_{GRIDWORLD}_n[{MAX_STEPS}]_alpha[{ALPHA}]_gamma[{GAMMA}]_batch[{BATCH_SIZE}]_weights[{WEIGHTS_METHOD}]_exploit[{EXPLOIT}]"
print(FILE_SIG)


# In[3]:


if GRIDWORLD == "WILLEMSEN":
    grid = np.ones((3, 9)) * -1
    grid[1, :8] = 0
    grid[1, 8] = 1
    grid[2, :8] = 0
    terminal_state = []
    for i in [0, 2]:
        for j in range(8):
            terminal_state.append((i, j))
    terminal_state.append((1, 8))
    terminal_reward = set(terminal_state)
    initial_state = (1, 0)
    blacked_state = {(0, 8), (2, 8)}
    # terminal_state = np.array(terminal_state)
    # initial_state = np.array([1, 0])
    # blacked_state = np.array([[0, 8], [2, 8]])
elif GRIDWORLD == "STRAIGHT":
    grid = np.ones((3, 9)) * -1
    grid[1, :8] = 0
    grid[1, 8] = 1
    grid[2, :8] = 0.1
    terminal_state = []
    for i in [0]:
        for j in range(8):
            terminal_state.append([i, j])
    terminal_state.append([1, 8])
    # terminal_state.append([0, 8])
    # terminal_state.append([2, 8])
    terminal_state = np.array(terminal_state)
    initial_state = np.array([1, 0])
    blacked_state = np.array([[0, 8], [2, 8]])
elif GRIDWORLD == "POOL":
    grid = np.ones((8, 8)) * 0
    grid[3:6, 3:6] = -1
    grid[7, 7] = 1
    terminal_state = np.array([7, 7])
    initial_state = np.array([1, 0])
    blacked_state = np.array([[5, 5], [4, 5], [5, 4], [4, 4]])
    # blacked_state = np.array([[np.nan, np.nan], [np.nan, np.nan]])
elif GRIDWORLD == "STAR":
    grid = np.ones((15, 15)) * 0.1
    grid[7, 0] = -1
    grid[14, 7] = -1
    grid[7, 14] = 1
    grid[0, 7] = 1
    terminal_state = np.array([[7,0], [14,7], [7,14], [0,7]])
    initial_state = np.array([7,7])
    blacked_state = np.array([np.nan, np.nan])

env = Gridworld(grid, terminal_state, initial_state, blacked_state)
env_greedy = Gridworld(grid, terminal_state, initial_state, blacked_state)

if GRIDWORLD == "STRAIGHT" or GRIDWORLD=="WILLEMSEN":
    states = [(i, j) for i in [0, 1, 2] for j in range(env.grid_width)]
    env_shape = (3, env.grid_width)
elif GRIDWORLD == "POOL" or GRIDWORLD == "STAR":
    states = [(i, j) for i in range(env.grid_height) for j in range(env.grid_width)]
    env_shape = (env.grid_height, env.grid_width)
else:
    print("WORLD not found")


# In[4]:


AGENTS = {
    "SIMPLE_Q": SimpleQ,
    "ME_Q": MEQ,
    "NOVELTOR": Noveltor,
    "RMAX": RMAXQ,
    }


# In[5]:


# MAIN TRAINING and EVALUATION LOOP
# os.mkdir('temp_epochs')
for n in tqdm(range(EPOCHS)):
    agent = AGENTS[AGENT_TYPE](dim2D=dim2D)
    rb = ReplayMemory(max_size=10000)
    env.reset()
    env_greedy.reset()
    metrics = eval_every_trajectory(MAX_STEPS, BATCH_SIZE, EPISODE_TIMEOUT, agent, rb, env, env_greedy, states, env_shape, EXPLOIT)
    # M.append(metrics)
    with open(f'{FOLDER}\\temp_epochs\\{n}_{FILE_SIG}.txt', 'wb') as fh:
        cPickle.dump(metrics, fh)


# # In[ ]:


# with open('%s.txt'%FILE_SIG, 'wb') as fh:
#     cPickle.dump(M, fh)

# # In[ ]:


# # percentile based on
# # http://www.jtrive.com/the-empirical-bootstrap-for-confidence-intervals-in-python.html

# # adapted according to empirical on
# # https://github.com/LizaLebedeva/bootstrap-experiments/blob/master/bootstrap_methods.ipynb

# def bootstrap_empirical(data, n=1000, func=np.mean):
#     """
#     Generate `n` bootstrap samples, evaluating `func`
#     at each resampling. `bootstrap` returns a function,
#     which can be called to obtain confidence intervals
#     of interest.
#     """
#     simulations = list()
#     sample_size = len(data)
#     for c in range(n):
#         itersample = np.random.choice(data, size=sample_size, replace=True)
#         simulations.append(func(itersample))
#     simulations.sort()
#     obs_metric = func(data) # compute CI center from data directly
# #     simulations = simulations - obs_metric # now move bootstrap to differences
#     def ci(p):
#         """
#         Return 2-sided symmetric confidence interval specified
#         by p.
#         """
#         u_pval = (1+p)/2.
#         l_pval = (1-u_pval)
#         l_indx = int(np.floor(n*l_pval))
#         u_indx = int(np.floor(n*u_pval))
#         percentile_upper = simulations[l_indx]
#         percentile_lower = simulations[u_indx]
#         empirical_lower = 2*obs_metric - percentile_lower
#         empirical_upper = 2*obs_metric - percentile_upper
#         return(empirical_lower,empirical_upper)
#     return(ci)


# # In[ ]:


# R = np.zeros((EPOCHS, MAX_STEPS, 2))

# for i, epoch in enumerate(M):
#     for j, metrics in enumerate(epoch):
#         for exp, g in enumerate(metrics['Gn']):
#             R[i,j, 0] += GAMMA**exp*g
#         for exp, g in enumerate(metrics['Gg']):
#             R[i,j, 1] += GAMMA**exp*g


# # In[ ]:


# Rn = R[:,:,0]
# Rg = R[:,:,1]
# yn = np.mean(Rn, axis=0)
# yg = np.mean(Rg, axis=0)
# yCIn = list(map(lambda p: bootstrap_empirical(p)(0.95), Rn.T))
# yCIg = list(map(lambda p: bootstrap_empirical(p)(0.95), Rg.T))
# Yn = np.transpose(yCIn)
# Yg = np.transpose(yCIg)


# # In[ ]:


# # get_ipython().run_line_magic('matplotlib', 'inline')
# x = np.arange(0, MAX_STEPS, 1)

# fig, ax = plt.subplots(figsize=(8,5))
# plt.plot(x, yn, label='exploring')
# ax.fill_between(x, Yn[0], Yn[1], color='teal', alpha=.2)

# plt.plot(x, yg, label='exploiting')
# ax.fill_between(x, Yg[0], Yg[1], color='purple', alpha=.2)

# plt.ylabel('cumulative discounted reward')
# plt.xlabel('episode')

# plt.legend()
# plt.savefig(f'results/{FILE_SIG}_exploit.png')
# plt.show()





# ## Loading Previously Pickled Files for plotting

# In[ ]:


# # load me q willemsen
# FILE_SIG = "epochs_ME_Q_WILLEMSEN_n[10000]_alpha[0.1]_gamma[0.9]_batch[10]_weights[exp_size_corrected]_exploit[True]"
# with open (f'results/{FILE_SIG}.txt', 'rb') as fh:
#     M = cPickle.load(fh)


# # In[ ]:


# # load simple q willemsen
# FILE_SIG = "epochs_SIMPLE_Q_WILLEMSEN_n[10000]_alpha[0.1]_gamma[0.9]_batch[10]_weights[exp_size_corrected]_exploit[True]"
# with open (f'results/{FILE_SIG}.txt', 'rb') as fh:
#     M = cPickle.load(fh)


# # In[ ]:


# M[90][1000].items()


# # In[ ]:




