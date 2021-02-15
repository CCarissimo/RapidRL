from functions import *
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# Create the gridworld, and set cell values to equal rewards
print('(Creating Gridworld)')
grid = np.ones((3, 9)) * -1
grid[1,:8] = 0
grid[1, 8] = 1
terminal_states = np.array([[1, 8]])
initial_state = np.array([1, 0])
blacked_state = np.array([[0, 8],[2, 8]])

print(grid)

n_episodes = 1000
max_steps = 30
gamma = 0.95

random_agent = agent()
env = gridworld(grid, terminal_states, initial_state, blacked_state, max_steps)

print('(Generating Trajectories)')
trajectories = []
reward_table = {}
visits = {}

for i in tqdm.tqdm(range(n_episodes)):
    state = env.reset()
    trajectory = []
    terminal = False

    discounted_reward = 0

    while not terminal:
        action = random_agent.next_action(state)

        trajectory.append((state, action))

        next_state, reward, terminal = env.step(action)

        if str(state) not in reward_table.keys():
            reward_table[str(state)] = {action: reward}
            visits[str(state)] = 1
        else:
            reward_table[str(state)][action] = reward
            visits[str(state)] += 1

        state = next_state

    trajectories.append(trajectory)

print('(Calculating Returns, Abstractions, Features, Visits)')
abstraction = {}
returns_table = return_table(trajectories, gamma, reward_table)

rho = state_dist(visits)

for action in env.actions:
    if action in abstraction.keys():
        pass
    else:
        abstraction[action] = Qpi_a(action, returns_table, rho)

sa_values = []
a_values = []
for sa, v in returns_table.items():
    sa_values.append(v)
    for a, v_abs in abstraction.items():
        if a in sa:
            a_values.append(v_abs)

bias_squared = [(sa_values[i] - a_values[i]) ** 2 for i in range(len(sa_values))]

bias2_hist_frame = np.zeros((4, 9))
counter = 0
for action, value in abstraction.items():
    for i in range(len(bias2_hist_frame[0, :])):
        if a_values[i] == value:
            bias2_hist_frame[counter, i] = (sa_values[i] - a_values[i]) ** 2
    counter += 1

heatmap = np.copy(grid)*0

for k, prob in rho.items():
    heatmap[int(k[1]), int(k[3])] = prob

# First plot
print('(Plotting 1 of 2)')
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))

ax[0, 0].scatter(sa_values, a_values)
ax[0, 0].set_title('Bias plot')
ax[0, 0].set_xlabel('state action pair value')
ax[0, 0].set_ylabel('action abstraction value')
ax[0, 0].set_ylim((0, 40))

ax[0, 1].hist(bias_squared, cumulative=True, bins=9)
ax[0, 1].set_title('Bias squared plot')
ax[0, 1].set_xlabel('(sa val - a val)**2')
ax[0, 1].set_ylabel('freq')
# plt.ylim((0, 40))

ax[1, 0].imshow(heatmap, cmap='binary')
ax[1, 0].set_title('gridworld heatmap')

ax[1, 1].hist(np.array(sa_values) - np.array(a_values))

plt.tight_layout()
plt.show()

# Second plot
print('(Plotting 2 of 2)')
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8), sharex=False, sharey=True)

plt.title('bias**2 cumulative density plot')

fig.add_subplot(111, frameon=False, visible=True, xticks=[], yticks=[])

colors = np.array([[60, 50, 255], [100, 50, 255], [140, 50, 255], [180, 50, 255]]) / 255

for i, action in enumerate(abstraction.keys()):
    ax[i // 2, i % 2].hist(bias2_hist_frame[i, :], bins=9, cumulative=True, color=colors[i])
    ax[i // 2, i % 2].set_title(action)

plt.xlabel(r'$(Q(s,a) - Q(a))^2$', labelpad=20)
plt.ylabel('freq', labelpad=20)

fig.suptitle(r'$bias^2$ cumulative distribution plot')

plt.tight_layout()
plt.show()