from classes import *

env = colored_MAB(3, sigma_A=1, sigma_C=1, initial_mu_C=[0, 0.5, 1])
agent = Agent(0.05, 0.05, 1, env)

agent.__init__run__()
agent.run(1000)
print(env.n_colors)
print(env.arm_distributions)