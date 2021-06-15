import numpy as np
from MAB.classes import *
import matplotlib.pyplot as plt

env = colored_MAB(n_colors=3)
agent = Agent(0.05, 0.99, env)

best_arm = agent.run(iterations=1000)

print(best_arm)
print(agent.N)
