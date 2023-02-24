"""
The purpose of this file is to run all the experiments for the paper we are preparing for ALIFE 2023
on death in Reinforcement Learning.

We will have an environment (possibly more) where we encode a new kind of state, a death state.
If agents reach the death state, rather than get reset to the initial state, like a normal terminal state,
they die.

When agents die, they are replaced by their progeny, their children, or other members of their culture,
at the initial state. A child is born, and may preserve some experiences of her ancestor at birth.
Specifically:
- a child may receive a complete Q-table from her ancestor
= a child may receive a partial Q-table from her ancestor
- a child may receive a complete replay buffer from her ancestor
- a child may receive a partial replay buffer from her ancestor

In these experiments, we will try to test all the different flavours of death and information preservation. We can
think of such processes as artificial culture creation, and the stories that we tell our children.

Finally, these agents will not be typical Q-learning agents. They will be exploring agents that learn with novelty values.
What this means is best explained in the paper. In a nutshell, agents seek to explore the environment, gathering novelty
rewards which reflect how new a particular visited state is. They use these novelty rewards as they would regular
rewards and update an online estimate of the expected sum of discounted future novelty rewards, in a Q-table, to make
their decisions. They follow an epsilon greedy policy with their novelty Q-table.

All calculated metrics are saved in a pandas dataframe for ease of data analysis.
- If you would like to calculate more things than are currently saved, simply add them to the results dictionary.
- You can see in Experiments.online_learning.online_learning what metrics are saved during trajectories
- Edit Experiments.online_learning.online_learning if you wish to save additional things
"""

import Agents
import Experiments
import Environments
import Utils
import numpy as np
import pandas as pd
import os

MAX_STEPS = 10000
EPISODE_TIMEOUT = 10000
GAMMA = 0.2
ALPHA = 0.1
BATCH_SIZE = 10
buffer_size_list = np.linspace(100, MAX_STEPS, 31).astype(int)
REPETITIONS = 40

cwd = os.getcwd()
FOLDER = "%s" % cwd
FILE_SIG = f"death_experiment_n[{MAX_STEPS}]_alpha[{ALPHA}]_gamma[{GAMMA}]_batch[{BATCH_SIZE}]"
print(FILE_SIG)

grid = np.zeros((15, 15))

np.random.seed(seed=0)
x_random_death_states = np.random.randint(1, 15, size=10)
np.random.seed(seed=1)
y_random_death_states = np.random.randint(1, 15, size=10)
terminal_state = set(zip(y_random_death_states, x_random_death_states))

initial_state = (0, 0)
blacked_state = {}

# UNCOMMENT IF YOU WOULD LIKE A PLOT OF THE GRIDWORLD BEFORE STARTING
_, _, _, _ = Utils.plot_gridworld(grid, terminal_state, initial_state, blacked_state)

master = []
storage = {}

for buffer_size in buffer_size_list:
    visits = []
    n_tables = []
    for iteration in range(REPETITIONS):
        env = Environments.Gridworld(grid, terminal_state, initial_state, blacked_state)
        states = [(i, j) for i in range(env.grid_height) for j in range(env.grid_width)]
        env_shape = (env.grid_height, env.grid_width)

        agent = Agents.SimpleNoveltor(ALPHA=ALPHA, GAMMA=GAMMA)
        rb = Agents.ReplayMemory(max_size=buffer_size)

        metrics, trajectory_metrics = Experiments.online_learning(MAX_STEPS, BATCH_SIZE, EPISODE_TIMEOUT, agent, rb,
                                                                  env, states, env_shape)

        trajectory_length_per_step = [metrics[t]["traj_len"] for t in range(MAX_STEPS)]
        lifetime_per_agent = [trajectory_metrics[t]["lifetime"] for t in range(len(trajectory_metrics))]

        # store final visits and n_tables
        # visits.append([[trajectory_metrics[n]['visits'][s] for s in states] for n in range(len(trajectory_metrics))])
        # n_tables.append([[trajectory_metrics[n]['n_table'][s] for s in states] for n in range(len(trajectory_metrics))])

        # intergenerational differences of visits (states been) and n-table (q-learning of novelties)
        visits_differences = [[trajectory_metrics[n + 1]['visits'][s] - trajectory_metrics[n]['visits'][s]
                               for s in states] for n in range(len(trajectory_metrics) - 1)]

        n_table_differences = [[(trajectory_metrics[n + 1]['n_table'][s] - trajectory_metrics[n]['n_table'][s]).mean()
                                for s in states] for n in range(len(trajectory_metrics) - 1)]

        abs_visits_differences = [[abs(trajectory_metrics[n + 1]['visits'][s] - trajectory_metrics[n]['visits'][s])
                                   for s in states] for n in range(len(trajectory_metrics) - 1)]

        abs_n_table_differences = [
            [np.abs(trajectory_metrics[n + 1]['n_table'][s] - trajectory_metrics[n]['n_table'][s]).mean()
             for s in states] for n in range(len(trajectory_metrics) - 1)]

        # print(n_table_differences)

        results = {
            "buffer_size": buffer_size,
            "repetition": iteration,
            "number_of_deaths": len(trajectory_metrics),
            "lifetime_average": np.mean(lifetime_per_agent),
            "lifetime_variance": np.var(lifetime_per_agent),
            "visits_differences_mean": np.mean(visits_differences),
            "abs_visits_differences_mean": np.mean(abs_visits_differences),
            "visits_differences_var": np.var(visits_differences),
            "n_table_differences_mean": np.mean(n_table_differences),
            "abs_n_table_differences_mean": np.mean(abs_n_table_differences),
            "n_table_differences_var": np.var(n_table_differences),
        }
        master.append(results)


final_df = pd.DataFrame(master)
print(final_df)
final_df.to_csv(FOLDER + "/" + FILE_SIG + ".csv")
