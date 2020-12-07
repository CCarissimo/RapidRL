from functions import *
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# Create the gridworld, and set cell values to equal rewards
print('(Creating Gridworld)')
grid = np.ones((3, 4))*-1
terminal_states = np.array([[2,0], [2,3]])
initial_state = np.array([0,1])
grid[2,3] = 100

n_episodes = 1000
max_steps = 30
gamma = 0.8
alpha = 0.1
epsilon = 0.95

Learn = True
batch_size = 64

random_agent = agent()
env = gridworld(grid, terminal_states, initial_state, random_agent, max_steps)
buffer = ReplayBuffer(max_size=10000, input_shape=[2], n_actions=1)

print('(Generating Trajectories and Learning Loop)')
trajectories = []
visits = {}
q_table = {}
for row in range(len(env.grid[:,0])):
    for col in range(len(env.grid[0,:])):
        visits[str(np.array([row, col]))] = 0
        q_table[str(np.array([row, col]))] = {action: 0 for action in env.actions}

for i in tqdm.tqdm(range(n_episodes)):
    state = env.reset()
    trajectory = []
    terminal = False
    
    discounted_reward = 0
    
    while not terminal:
        action = random_agent.next_action(state)
        
#         rand = np.random.random()
#         max_action = max(q_table[str(state)], key=q_table[str(state)].get)
#         if rand < epsilon:
#             action = max_action
#         else:
#             action_to_avoid = action
#             action = np.random.choice(list(q_table[str(state)].keys()))
#             while action == action_to_avoid:
#                 action = np.random.choice(list(q_table[str(state)].keys()))
        
        trajectory.append((state, action))
        
        next_state, reward, terminal = env.step(action)
        
        action = env.actions_map[action]
        buffer.store_transition(state, action, reward, next_state, terminal)
        
        if Learn:
            if buffer.mem_cntr > batch_size:
                states, actions, rewards, states_, dones = buffer.sample_buffer(batch_size)
                
                for index in range(batch_size):
                    state = np.array([int(x) for x in states[index]])
                    action = env.actions[int(actions[index][0])]

                    q_table[str(state)][action] = q_table[str(state)][action] + alpha*(rewards[index] +                 gamma*max(v for v in q_table[str(state)].values()) - q_table[str(state)][action])

        visits[str(state)] += 1
        
        state = next_state
        
    trajectories.append(trajectory)

print('(Calculating Returns, Abstractions, Features, Visits, Biases)')
abstraction = {}

returns_table = return_table(trajectories, gamma, q_table)

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
            
bias_squared = [(sa_values[i] - a_values[i])**2 for i in range(len(sa_values))]

heatmap = np.zeros((3, 4))
heatmapQ = np.zeros((3, 4))

for k, prob in rho.items():
    heatmap[int(k[1]), int(k[3])] = prob

for state, actions in q_table.items():
    heatmapQ[int(state[1]), int(state[3])] = max(qval for qval in actions.values())
    
# First plot
print('(Plotting 1 of 1)')
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))

ax[0,0].scatter(sa_values, a_values)
ax[0,0].set_title('Bias plot')
ax[0,0].set_xlabel('state action pair value')
ax[0,0].set_ylabel('action abstraction value')
ax[0,0].set_ylim((0, 40))

ax[0,1].hist(bias_squared, cumulative=True, bins=9)
ax[0,1].set_title('Bias squared plot')
ax[0,1].set_xlabel('(sa val - a val)**2')
ax[0,1].set_ylabel('freq')
#plt.ylim((0, 40))

ax[1,0].imshow(heatmap, cmap='binary')
ax[1,0].set_title('gridworld visits heatmap')

ax[1,1].imshow(heatmapQ, cmap='binary')
ax[1,1].set_title('gridworld qvalues')

plt.tight_layout()
plt.show()