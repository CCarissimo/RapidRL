import numpy as np
from classes import *
import matplotlib.pyplot as plt
import tqdm

savepath = 'results\\'
name = 'avg_delta_naive_vs_oracle'

repetitions = 100
max_iter = 1000
epsilons = [0.05]
Deltas = np.zeros((2, len(epsilons), repetitions, max_iter))
color_means = [1]

for k, epsilon in enumerate(epsilons):
    for i in tqdm.tqdm(range(repetitions)):
        env = colored_MAB(n_colors=3)

        agent = Agent(0.05, epsilon, env)
        agent.__init__run__()
        for j in range(max_iter):
            agent.run(iterations=1)
            best_agent_arm = agent.recommend()
            arm_means = {key: env.arm_distributions[key].mu for key in env.arm_distributions.keys()}
            best_env_arm = max(arm_means, key=arm_means.get)
            oracle_delta = 1 - arm_means[best_env_arm]
            agent_delta = 1 - arm_means[best_agent_arm]
            Deltas[0, k, i, j] = oracle_delta
            Deltas[1, k, i, j] = agent_delta

delta_averages = np.mean(Deltas, axis=2)

np.save('%s%s.npy' % (savepath, name), delta_averages)

print('Done with exp %s%s.npy' % (savepath, name))