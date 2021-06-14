import numpy as np
from classes import colored_MAB, Agent
import tqdm


savepath = 'results\\'
name = 'simple_regret_over_epsilon_x_variance'

search_params = np.linspace(0, 1, 51)
variances = np.linspace(0, 1, 51)
repetitions = 100
delta_averages = np.zeros((len(search_params), len(variances)))
initial_mu_C = [0, 0.5, 1]

for j, epsilon in tqdm.tqdm(enumerate(search_params)):
    for k, sigma in enumerate(variances):
        deltas = np.zeros(repetitions)
        for i in range(repetitions):
            env = colored_MAB(n_colors=3, sigma_A=1-sigma, sigma_C=sigma, initial_mu_C=initial_mu_C)

            agent = Agent(0.05, epsilon, 0, env)
            agent.__init__run__()
            agent.run(iterations=1000)
            best_agent_arm = agent.recommend()
            arm_means = {key: env.arm_distributions[key].mu for key in env.arm_distributions.keys()}
            best_env_arm = max(arm_means, key=arm_means.get)
            deltas[i] = max(initial_mu_C) + sigma/2 - arm_means[best_agent_arm]

        delta_averages[j, k] = np.mean(deltas)

np.save('%s%s.npy' % (savepath, name), delta_averages)
