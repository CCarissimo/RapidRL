import numpy as np
from classes import *
import matplotlib.pyplot as plt
import tqdm

savepath = 'results\\'

repetitions = 100
max_iter = 1000
epsilons = 1 - np.array([0.5, 0.75, 0.95, 0.98, 0.99, 0.999])
suboptimalities = np.zeros((len(epsilons), repetitions, max_iter))
arm_counts = np.zeros((len(epsilons), repetitions, max_iter))

for k, epsilon in enumerate(epsilons):
    for i in range(repetitions):
        env = colored_MAB(n_colors=3)
        agent = Agent(0.05, epsilon, env)
        agent.__init__run__()
        for j in range(max_iter):
            agent.run(iterations=1)
            best_agent_arm = agent.recommend()
            arm_means = {key: env.arm_distributions[key].mu for key in env.arm_distributions.keys()}
            #best_env_arm = max(arm_means, key=arm_means.get)
            suboptimality_gap = 1 - arm_means[best_agent_arm]
            suboptimalities[k, i, j] = suboptimality_gap
            arm_counts[k, i, j] = len(agent.Q.keys())

delta_averages = np.mean(suboptimalities, axis=(1))
arm_counts = np.mean(arm_counts, axis=1)

np.save('%savg_delta_naive_n&epsilon.npy'%(savepath), delta_averages)
np.save('%savg_arms_naive_n&epsilon.npy'%(savepath), arm_counts)

print('Done with exp naive n&epsilon')