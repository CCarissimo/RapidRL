import numpy as np
from classes import colored_MAB, Agent
import matplotlib.pyplot as plt

savepath = 'results\\'

search_params = np.concatenate((np.linspace(0, 0.1, 100), np.linspace(0.11, 1, 100)))
repetitions = 100
delta_averages = np.zeros((len(search_params)))
# color_means = [1]

for j, epsilon in enumerate(search_params):
    suboptimalities = np.zeros(repetitions)
    for i in range(repetitions):
        env = colored_MAB(n_colors=1)

        # # set the color distributions to be deterministic
        # for i, dist in enumerate(env.color_distributions):
        #     dist.a = color_means[i]
        #     dist.b = color_means[i]

        agent = Agent(0.05, epsilon, env)
        agent.__init__run__()
        agent.run(iterations=1000)
        best_agent_arm = agent.recommend()

        arm_means = {key: env.arm_distributions[key].mu for key in env.arm_distributions.keys()}
        # best_env_arm = max(arm_means, key=arm_means.get)
        suboptimality_gap = 1 - arm_means[best_agent_arm]
        suboptimalities[i] = suboptimality_gap

    delta_averages[j] = np.mean(suboptimalities)

np.save('%ssimple_regret_over_epsilon.npy' % (savepath), delta_averages)
