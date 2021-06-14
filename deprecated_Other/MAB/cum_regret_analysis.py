from classes import *
import tqdm

savepath = 'results\\'
deterministic = 'neither'
n_colors = 1
name = 'det_' + deterministic + '_cum_regret_oracle' + '_%i_colors' % n_colors

repetitions = 100
max_iter = 1000
epsilons = [0.05]
Deltas = np.zeros((2, len(epsilons), repetitions, max_iter))
color_means = np.linspace(0, 1, n_colors)

if deterministic == 'arms':
    color_spread = 0
    sigma = 1
elif deterministic == 'colors':
    color_spread = 0.5
    sigma = 0
else:
    color_spread = 0.5
    sigma = 1

max_arm_mean = max(color_means) + color_spread

for k, epsilon in enumerate(epsilons):
    for i in tqdm.tqdm(range(repetitions)):
        env = colored_MAB(n_colors=n_colors)

        for c, dist in enumerate(env.color_distributions):
            dist.a = color_means[c] - color_spread
            dist.b = color_means[c] + color_spread

        env.sigma = sigma

        agent = Agent(0.05, epsilon, env)
        agent.__init__run__()

        for j in range(max_iter):
            agent.run(iterations=1)
            agent_X = agent.result
            arm_means = {key: env.arm_distributions[key].mu for key in env.arm_distributions.keys()}
            best_env_arm = max(arm_means, key=arm_means.get)
            oracle_delta = max_arm_mean - arm_means[best_env_arm]
            agent_delta = max_arm_mean - agent_X
            Deltas[0, k, i, j] = oracle_delta
            Deltas[1, k, i, j] = agent_delta

delta_averages = np.mean(Deltas, axis=2)

np.save('%s%s.npy' % (savepath, name), delta_averages)

print('Done with exp %s%s.npy' % (savepath, name))
