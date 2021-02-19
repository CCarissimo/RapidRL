import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


# For (s,a), find the cumulative discounted sum of rewards over a set of trajectories
def Qpi_sa(state, action, trajectories, gamma, q_table):
    returns = []
    for trajectory in trajectories:
        store = False
        step_counter = 0
        for t in trajectory:
            disc_ret = 0
            if state == t.state and action == t.action:
                store = True

            if store:
                disc_ret += gamma ** step_counter * q_table[t.state][t.action]
                step_counter += 1

        if store:
            returns.append(disc_ret)

    return np.mean(returns)


# For (s,a), find the cumulative discounted sum of novelties over a set of trajectories
def Npi_sa(state, action, trajectories, gamma, visits):
    returns = []
    for trajectory in trajectories:
        store = False
        step_counter = 0
        for t in trajectory:
            disc_ret = 0
            if state == t.state and action == t.action:
                store = True

            if store:
                disc_ret += gamma ** step_counter * visits[t.state][t.action]
                step_counter += 1

        if store:
            returns.append(disc_ret)

    return np.mean(returns)


# Create a dictionary indexed by state action strings with discounted
def cumulative_table(trajectories, gamma, function_sa, table):
    returns_table = {}
    for trajectory in trajectories:
        for t in trajectory:
            if t.state in returns_table.keys():
                if t.action in returns_table[t.state].keys():
                    continue
                else:
                    returns_table[t.state][t.action] = function_sa(t.state, t.action, trajectories, gamma, table)
            else:
                returns_table[t.state] = {t.action: function_sa(t.state, t.action, trajectories, gamma, table)}

    return returns_table


# Calculate state visit distribution
def state_dist(visits):
    total_visits = sum(v for state,v in visits.items())
    rho = {}
    for s in visits.keys():
        rho[s] = visits[s] / total_visits
    return rho


# Using the table of returns for (s,a) pairs, abstract over states to estimate generalized (a) values
def Qpi_a(action, return_table, rho):
    abstraction = 0
    for state, actions in return_table.items():
        if action in actions.keys():
            abstraction += rho[state] * return_table[state][action]
    return abstraction


def plot_gridworld(grid, terminal_state, initial_state, blacked_state, fig=None, ax=None, show=True):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        fig = fig
        ax = ax

    cax = fig.add_axes([0.27, 0.75, 0.5, 0.05])
    colormap = plt.get_cmap('RdYlGn')
    im = ax.imshow(grid, cmap=colormap)
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which='minor', color='w', linewidth=2)
    cb = fig.colorbar(im, ax=ax, cax=cax, orientation='horizontal', label='reward')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(labelsize=10)
    for i in range(len(grid[:, 0])):
        for j in range(len(grid[0, :])):
            blacked_bool = (np.array([i, j]) == blacked_state).all(1)
            initial_bool = (np.array([i, j]) == initial_state).all(0)
            terminal_bool = (np.array([i, j]) == terminal_state).all(1)
            if blacked_bool.any():
                ax.text(j, i, 'X', ha="center", va="center", color="w", fontsize=20)
            elif initial_bool.any():
                ax.text(j, i, 'S', ha="center", va="bottom", color="black", position=(j + 0.25, i - 0.15))
                ax.text(j, i, grid[i, j], ha="center", va="center", color="w")
            elif terminal_bool.any():
                ax.text(j, i, 'T', ha="center", va="bottom", color="black", position=(j + 0.25, i - 0.15))
                ax.text(j, i, grid[i, j], ha="center", va="center", color="w")
            else:
                ax.text(j, i, grid[i, j], ha="center", va="center", color="w")
    if show:
        plt.show()

    return fig, ax, im, cb


def run_trajectory(env, agent, epsilon, abstract=False):
    env.reset()
    agent.reset_trajectory()
    agent.epsilon = epsilon
    agent.abstract = abstract

    while not env.terminal:
        action = agent.select_action(env.state)

        transition = env.step(action)

        agent.observe(transition)

    return agent.trajectory


def generate_codes(verts):
    codes = []
    if len(verts) >= 2:
        codes.append(Path.MOVETO)
        for i in range(1, len(verts)):
            codes.append(Path.LINETO)
        return (codes)
    else:
        print('trajectory less than 2 steps')


def plot_trajectory(trajectory, grid, terminal_state, initial_state, blacked_state, show=True, fig=None, ax=None):
    verts = [np.flip(t.state) for t in trajectory]
    codes = generate_codes(verts)
    path = Path(verts, codes)

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    patch = patches.PathPatch(path, facecolor='none', lw=2)
    ax.add_patch(patch)

    fig, ax, im, cb = plot_gridworld(grid, terminal_state, initial_state, blacked_state, fig=fig, ax=ax, show=show)

    if show:
        pass
    else:
        return fig, ax, im, cb, patch


def action_abstraction(sa_values_table, state_dist_table):
    states = list(sa_values_table.keys())
    arbitrary_state = states[0]
    action_keys = sa_values_table[arbitrary_state].keys()
    abstraction = {}
    for action in action_keys:
        if action in abstraction.keys():
            pass
        else:
            abstraction[action] = Qpi_a(action, sa_values_table, state_dist_table)

    return abstraction


def action_abstraction_bias(sa_values_table, a_abstraction_table):
    sa_values = []
    a_values = []
    for state, actions in sa_values_table.items():
        for a, v_a in actions.items():
            sa_values.append(v_a)
            a_values.append(a_abstraction_table[a])

    bias_squared = [(sa_values[i] - a_values[i]) ** 2 for i in range(len(sa_values))]

    return sa_values, a_values, bias_squared


def generate_heatmap(grid, table, aggf=None):
    hm = np.copy(grid) * 0
    if aggf is None:
        aggf = lambda x: x

    for k, v in table.items():
        hm[k] = aggf(v)

    return hm
