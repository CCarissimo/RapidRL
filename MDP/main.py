import numpy as np
import MDP
from warnings import filterwarnings
import matplotlib.pyplot as plt

GRIDWORLD = "POOL"
AGENT_TYPE = "LINEAR_RMAX"
ANIMATE = False
MAX_STEPS = 1000
EPISODE_TIMEOUT = 32
GAMMA = 0.8
ALPHA = 0.1
BATCH_SIZE = 1
WEIGHTS_METHOD = "exponential"

FILE_SIG = f"{AGENT_TYPE}_{GRIDWORLD}_n[{MAX_STEPS}]_alpha[{ALPHA}]_gamma[{GAMMA}]_batch[{BATCH_SIZE}]_weights[{WEIGHTS_METHOD}]"
print(FILE_SIG)
if GRIDWORLD == "WILLEMSEN":
    grid = np.ones((3, 9)) * -1
    grid[1, :8] = 0
    grid[1, 8] = 1
    grid[2, :8] = 0
    terminal_state = []
    for i in [0, 2]:
        for j in range(8):
            terminal_state.append([i, j])
    terminal_state.append([1, 8])
    terminal_state = np.array(terminal_state)
    initial_state = np.array([1, 0])
    blacked_state = np.array([[0, 8], [2, 8]])
elif GRIDWORLD == "POOL":
    grid = np.ones((8, 8)) * 0
    grid[3:6, 3:6] = -1
    grid[7, 7] = 1
    terminal_state = np.array([7, 7])
    initial_state = np.array([1, 0])
    # blacked_state = np.array([[5, 5], [4, 5], [5, 4], [4, 4]])
    blacked_state = np.array([[np.nan, np.nan], [np.nan, np.nan]])

# _, _, _, _ = MDP.plot_gridworld(grid, terminal_state, initial_state, blacked_state)

env = MDP.Gridworld(grid, terminal_state, initial_state, blacked_state, EPISODE_TIMEOUT)
env_greedy = MDP.Gridworld(grid, terminal_state, initial_state, blacked_state, EPISODE_TIMEOUT)

if GRIDWORLD == "WILLEMSEN":
    states = [(1, j) for j in range(env.grid_width)]
    env_shape = (1, env.grid_width)
elif GRIDWORLD == "POOL":
    states = [(i, j) for i in range(env.grid_height) for j in range(env.grid_width)]
    env_shape = (env.grid_height, env.grid_width)
else:
    print("WORLD not found")

filterwarnings('ignore')

if AGENT_TYPE == 'NOVELTOR':
    class ME_AIC_Learner:
        def __init__(self):
            if GRIDWORLD == 'WILLEMSEN':
                self.Qs = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.identity())
                self.Qg = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.global_context())
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)
            elif GRIDWORLD == 'POOL':
                self.Qs = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.identity())
                self.Qg = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.global_context())
                self.Qc = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.column())
                self.Qr = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.row())
                # self.Ql = MDP.LinearNoveltyEstimator(alpha=ALPHA, gamma=GAMMA)
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg, self.Qr, self.Qc], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)

        def select_action(self, t, greedy=False):
            if t.action != 'initialize':
                reward = 1/(self.Qs.visits[t.state][t.action] + 1) if t.state_ != t.state else 0
                self.Qe.update_RSS(t.action, reward, t.state_)
            if greedy:  # use estimator with minimum RSS
                values = self.Qe.predict(t.state)
                # print(t, values)
                return np.random.choice(np.flatnonzero(values == values.max()))
            else:  # use the AIC combined estimators
                values = self.Qe.predict(t.state)
                return np.random.choice(np.flatnonzero(values == values.max()))

        def update(self, transitions):
            for t in transitions:
                self.Qe.update(t)
elif AGENT_TYPE == 'LINEAR_NOVELTOR':
    class ME_AIC_Learner:
        def __init__(self):
            if GRIDWORLD == 'WILLEMSEN':
                self.Qs = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.identity())
                self.Qg = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.global_context())
                self.Ql = MDP.LinearNoveltyEstimator(alpha=ALPHA, gamma=GAMMA)
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg, self.Ql], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)
            elif GRIDWORLD == 'POOL':
                self.Qs = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.identity())
                self.Qg = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.global_context())
                self.Qc = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.column())
                self.Qr = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.row())
                # self.Ql = MDP.LinearNoveltyEstimator(alpha=ALPHA, gamma=GAMMA)
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg, self.Qr, self.Qc], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)

        def select_action(self, t, greedy=False):
            if t.action != 'initialize':
                reward = 1/(self.Qs.visits[t.state][t.action] + 1) if t.state_ != t.state else 0
                self.Qe.update_RSS(t.action, reward, t.state_)
            if greedy:  # use estimator with minimum RSS
                values = self.Qe.predict(t.state)
                # print(t, values)
                return np.random.choice(np.flatnonzero(values == values.max()))
            else:  # use the AIC combined estimators
                values = self.Qe.predict(t.state)
                return np.random.choice(np.flatnonzero(values == values.max()))

        def update(self, transitions):
            for t in transitions:
                self.Qe.update(t)

elif AGENT_TYPE == 'PSEUDOCOUNT':
    class ME_AIC_Learner:
        def __init__(self):
            self.N = MDP.pseudoCountNovelty(features=[MDP.identity, MDP.row, MDP.column], alpha=1)
            if GRIDWORLD == 'WILLEMSEN':
                self.Qs = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.identity())
                self.Qg = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.global_context())
                self.Ql = MDP.LinearNoveltyEstimator(alpha=ALPHA, gamma=GAMMA)
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg, self.Ql], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)
            elif GRIDWORLD == 'POOL':
                self.Qs = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.identity())
                self.Qg = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.global_context())
                self.Qc = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.column())
                self.Qr = MDP.N_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.row())
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg, self.Qr, self.Qc], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)

        def select_action(self, t, greedy=False):
            if t.action != 'initialize':
                novelty = self.N.evaluate(t.state) if t.state_ != t.state else 0
                self.Qe.update_RSS(t.action, novelty, t.state_)
            if greedy: # use estimator with minimum RSS
                values = self.Qe.predict(t.state)
                # print(t, values)
                return np.random.choice(np.flatnonzero(values == values.max()))
            else: # use the AIC combined estimators
                values = self.Qe.predict(t.state)
                return np.random.choice(np.flatnonzero(values == values.max()))

        def update(self, transitions):
            for t in transitions:
                self.Qe.update(t)

elif AGENT_TYPE == "LINEAR":
    class ME_AIC_Learner:
        def __init__(self):
            if GRIDWORLD == 'WILLEMSEN':
                self.Qs = MDP.Q_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.identity())
                self.Ql = MDP.LinearEstimator(alpha=ALPHA, gamma=GAMMA)
                self.Qe = MDP.CombinedAIC([self.Qs, self.Ql], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)
            elif GRIDWORLD == 'POOL':
                self.Qs = MDP.Q_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.identity())
                self.Qg = MDP.Q_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.global_context())
                self.Qc = MDP.Q_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.column())
                self.Qr = MDP.Q_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.row())
                self.Ql = MDP.LinearEstimator(alpha=ALPHA, gamma=GAMMA)
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg, self.Qr, self.Qc, self.Ql], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)

        def select_action(self, t, greedy=False):
            if t.action != 'initialize':
                self.Qe.update_RSS(t.action, t.reward, t.state_)

            if greedy: # use estimator with minimum RSS
                values = self.Qe.predict(t.state)
                return np.random.choice(np.flatnonzero(values == values.max()))
            else: # use the AIC combined estimators
                values = self.Qe.predict(t.state)
                return np.random.choice(np.flatnonzero(values == values.max()))

        def update(self, transitions):
            for t in transitions:
                self.Qe.update(t)

elif AGENT_TYPE == "VANILLA":
    class ME_AIC_Learner:
        def __init__(self):
            if GRIDWORLD == 'WILLEMSEN':
                self.Qs = MDP.Q_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.identity())
                self.Qg = MDP.Q_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.global_context())
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)
            elif GRIDWORLD == 'POOL':
                self.Qs = MDP.Q_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.identity())
                self.Qg = MDP.Q_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.global_context())
                self.Qc = MDP.Q_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.column())
                self.Qr = MDP.Q_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.row())
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg, self.Qr, self.Qc], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)

        def select_action(self, t, greedy=False):
            if t.action != 'initialize':
                self.Qe.update_RSS(t.action, t.reward, t.state_)

            if greedy: # use estimator with minimum RSS
                values = self.Qe.predict(t.state)
                return np.random.choice(np.flatnonzero(values == values.max()))
            else: # use the AIC combined estimators
                values = self.Qe.predict(t.state)
                return np.random.choice(np.flatnonzero(values == values.max()))

        def update(self, transitions):
            for t in transitions:
                self.Qe.update(t)

elif AGENT_TYPE == "RMAX":
    class ME_AIC_Learner:
        def __init__(self):
            if GRIDWORLD == 'WILLEMSEN':
                self.Qs = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.1, mask=MDP.identity())
                self.Qg = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.1, mask=MDP.global_context())
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)
            elif GRIDWORLD == 'POOL':
                self.Qs = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.identity())
                self.Qg = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.global_context())
                self.Qc = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.column())
                self.Qr = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.row())
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg, self.Qr, self.Qc], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)

        def select_action(self, t, greedy=False):
            if t.action != 'initialize':
                self.Qe.update_RSS(t.action, t.reward, t.state_)

            if greedy: # use estimator with minimum RSS
                values = self.Qe.predict(t.state)
                return np.random.choice(np.flatnonzero(values == values.max()))
            else: # use the AIC combined estimators
                values = self.Qe.predict(t.state)
                return np.random.choice(np.flatnonzero(values == values.max()))

        def update(self, transitions):
            for t in transitions:
                self.Qe.update(t)

elif AGENT_TYPE == "LINEAR_RMAX":
    class ME_AIC_Learner:
        def __init__(self):
            if GRIDWORLD == 'WILLEMSEN':
                self.Qs = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.5, mask=MDP.identity())
                self.Qg = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.5, mask=MDP.global_context())
                self.Ql = MDP.LinearEstimator(alpha=ALPHA, gamma=GAMMA, b=0.5)
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)
            elif GRIDWORLD == 'POOL':
                self.Qs = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.5, mask=MDP.identity())
                self.Qg = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.5, mask=MDP.global_context())
                self.Qc = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.5, mask=MDP.column())
                self.Qr = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.5, mask=MDP.row())
                self.Ql = MDP.LinearEstimator(alpha=ALPHA, gamma=GAMMA, b=0.5)
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg, self.Qr, self.Qc, self.Ql], RSS_alpha=0.9,
                                          weights_method=WEIGHTS_METHOD)

        def select_action(self, t, greedy=False):
            if t.action != 'initialize':
                self.Qe.update_RSS(t.action, t.reward, t.state_)

            if greedy:  # use estimator with minimum RSS
                values = self.Qe.predict(t.state)
                return np.random.choice(np.flatnonzero(values == values.max()))
            else:  # use the AIC combined estimators
                values = self.Qe.predict(t.state)
                return np.random.choice(np.flatnonzero(values == values.max()))

        def update(self, transitions):
            for t in transitions:
                self.Qe.update(t)
else:
    print(AGENT_TYPE, "not found")


agent = ME_AIC_Learner()
rb = MDP.ReplayMemory(max_size=10000)

trajectory = []
trajectories = []
metrics = []
epilen = []
Gn = []
step = 0

# MAIN TRAINING and EVALUATION LOOP
for i in range(MAX_STEPS):
    # EXPLORE
    action = agent.select_action(env.transition)
    transition = env.step(action)
    rb.append(transition)
    Gn.append(transition.reward)
    S = rb.sample(batch_size=BATCH_SIZE)
    agent.update(S)
    trajectory.append(transition)
    print(transition)
    # EXPLOIT: by setting action selection to be exploitative: "greedy"
    Gg = []
    # RUN entire trajectory, and set greedy env to the initial state
    env_greedy.reset()
    while not env_greedy.terminal and not env_greedy.timeout:
        action = agent.select_action(env_greedy.transition, greedy=True)
        transition = env_greedy.step(action)
        Gg.append(transition.reward)

    step += 1

    # Q_matrix = np.zeros((env.grid_height, env.grid_width, 4))
    Q_matrix = [agent.Qe.predict(s) for s in states]

    V_vector = [max(q) for q in Q_matrix]
    # print(V_vector)
    imV = np.reshape(V_vector, (env_shape[0], env_shape[1])).T
    A_vector = [np.argmax(q) for q in Q_matrix]
    imA = np.reshape(A_vector, (env_shape[0], env_shape[1])).T

    K = [e.count_parameters() for e in agent.Qe.estimators]
    RSS = agent.Qe.RSS

    # print(imV)

    metrics.append({
        't': i,
        'V': imV,
        'A': imA,
        'S': transition.state_,
        'Gn': Gn,
        'Gg': Gg,
        'W': agent.Qe.W,
        'K': K,
        'RSS': RSS,
        'steps': step
    })

    if env.terminal or env.timeout:
        trajectories.append(trajectory)
        epilen.append([len(Gn), len(Gg)])

        env.reset()
        trajectory = []
        Gn = []

# PLOT Cumulative Returns over time #

x = np.arange(0, metrics[-1]['steps'])
yn = []
yg = []
for i, metric in enumerate(metrics):
    Yn = 0
    Yg = 0
    for j, g in enumerate(metric['Gn']):
        Yn += GAMMA ** j * g
    for j, g in enumerate(metric['Gg']):
        Yg += GAMMA ** j * g

    yn.append(Yn)
    yg.append(Yg)

cumEpilen = np.cumsum([ele[0] for ele in epilen])
cumEpiG = np.cumsum([ele[1] for ele in epilen])

# plt.figure(0)
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 5))
ax1.plot(x, yn, label='exploring')

ax1.plot(x, yg, label='exploiting')

ax1.set_ylabel('cumulative discounted reward')
MDP.plt.xlabel('steps')

ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

ax2.set_ylabel('trajectory length')
ax2.scatter(cumEpilen, [ele[0] for ele in epilen], s=5)
ax2.scatter(cumEpiG, [ele[1] for ele in epilen], s=5)
MDP.plt.tight_layout()
plt.savefig(f'{AGENT_TYPE}\\G_{FILE_SIG}.png')

# PLOT Estimator Evolution over time #

if 'LINEAR' in AGENT_TYPE:
    labels = [type(e.mask).__name__ for e in agent.Qe.estimators[:-1]] + ['linear']
else:
    labels = [type(e.mask).__name__ for e in agent.Qe.estimators]

plt.figure(2)
W = []
for i in range(len(agent.Qe.estimators)):
    w = [ele['W'][i] for ele in metrics]
    W.append(w)
    plt.plot(w, label=labels[i])
plt.title('W over time')
plt.legend()
plt.savefig(f'{AGENT_TYPE}\\W_{FILE_SIG}.png')

plt.figure(3)
K = []
for i in range(len(agent.Qe.estimators)):
    k = [ele['K'][i] for ele in metrics]
    K.append(k)
    plt.plot(k, label=labels[i])
plt.title('K over time')
plt.legend()
plt.savefig(f'{AGENT_TYPE}\\K_{FILE_SIG}.png')

plt.figure(4)
for i in range(len(agent.Qe.estimators)):
    RSS = [ele['RSS'][i] for ele in metrics]
    plt.plot(RSS, label=labels[i])
plt.title('RSS over time')
plt.legend()
plt.savefig(f'{AGENT_TYPE}\\RSS_{FILE_SIG}.png')


# Complexity PLOT
plt.figure(5)
W = np.array(W)
K = np.array(K)
C = (W * K).sum(axis=0)  # matrix product
plt.title('Complexity overf ftfifme')
plt.plot(C)
plt.savefig(f'{AGENT_TYPE}\\totK_{FILE_SIG}.png')


def overlay_actions(A):
    global ann_list
    k = 0
    for i in range(env.grid_width):
        for j in range(env.grid_height):
            if [j, i] in env.terminal_states:
                terminal_reward = env.grid[j, i]
                if len(ann_list) > k:
                    ann_list[k].set_text(f"{terminal_reward:.1f}".lstrip('0'))
                    k += 1
                else:
                    text = ax.text(j, i, f"{terminal_reward:.1f}".lstrip('0'),
                                   ha="center", va="center", color="w")
                    ann_list.append(text)
                    k += 1
            else:
                if len(ann_list) > k:
                    ann_list[k].set_text(env.actions[A[i, j]][0:1])
                    k += 1
                else:
                    text = ax.text(j, i, env.actions[A[i, j]][0:1],
                                   ha="center", va="center", color="w")
                    ann_list.append(text)
                    k += 1


def evaluate(ag, Env, n_episodes, greedy=True):
    """ Computes average undiscounted return from initial state without exploration. """
    T = []
    G = []
    for e in range(n_episodes):
        s = Env.reset()
        tau = [] # trajectory
        terminal, g, t = False, 0, 0
        while not terminal and t < EPISODE_TIMEOUT:
            a = ag.select_action(s, greedy=greedy)
            ns, r, terminal = Env.transition(s, a)
            tau.append((s, a, r, ns, terminal))
            g += r
            t += 1
            s = ns
        G.append(g)
        T.append(tau)
    return T, G


# Value Function Plot

print(metrics[-1]['V'])
print(metrics[-1]['A'])
# plt.figure(5)
fig, ax = plt.subplots()
plt.title('Value Function')
im = ax.imshow(metrics[-1]['V'], origin='lower')
ann_list = []
overlay_actions(metrics[-1]['A'])
plt.axis('off')
plt.savefig(f'{AGENT_TYPE}\\V_{FILE_SIG}.png')


visits = agent.Qs.visits
hm = MDP.generate_heatmap(grid=env.grid, table=visits, aggf=lambda s: np.sum(s))
plt.figure(7)
plt.title(r"Updates Heatmap $\approx$ Visits")
plt.imshow(hm, origin='lower')
plt.savefig(f'{AGENT_TYPE}\\Visits_{FILE_SIG}.png')
plt.show()

# ANIMATION

# Visualise Q-learning
if ANIMATE:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import animation, rc

    # VISUALISE_METRIC = "N"
    VISUALISE_METRIC = "V"
    # VISUALISE_METRIC = "AD"

    ann_list = []

    fig = plt.figure(figsize=(env.grid_width, env.grid_height))
    ax = fig.add_subplot(111)

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad="3%")

    ax.axis('off')
    ims = []

    tx = ax.set_title('')
    plt.tight_layout()
    fig.subplots_adjust(right=.8)


    def animate(i):
        im = ax.imshow(metrics[i][VISUALISE_METRIC], animated=True, origin='lower')
        if VISUALISE_METRIC == "V":
            overlay_actions(metrics[i]['A'])
        cax.cla()
        cb = fig.colorbar(im, cax=cax)
        tx.set_text(f'{VISUALISE_METRIC} at {metrics[i]["steps"] / MAX_STEPS * 100:2.0f}% training')


    ani = animation.FuncAnimation(fig, animate, frames=len(metrics))

    plt.close()

    # HTML(ani.to_jshtml())
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Michael Kaisers'), bitrate=100)
    ani.save(f"V_animation_{FILE_SIG}_-{VISUALISE_METRIC}.mp4", writer=writer)  # extra_args=['-vcodec', 'libx264']
