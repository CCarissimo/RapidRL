import numpy as np
import MDP
from warnings import filterwarnings
import matplotlib.pyplot as plt

GRIDWORLD = "WILLEMSEN"
AGENT_TYPE = "NOVELTOR"
ANIMATE = False
MAX_STEPS = 100
EPISODE_TIMEOUT = 34
GAMMA = 0.8
ALPHA = 0.1
BATCH_SIZE = 1
WEIGHTS_METHOD = "exponential"

FILE_SIG = f"{AGENT_TYPE}_{GRIDWORLD}_n[{MAX_STEPS}]_alpha[{ALPHA}]_gamma[{GAMMA}]_batch[{BATCH_SIZE}]"

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

states = [(1, j) for j in range(env.grid_width)]

filterwarnings('ignore')

if AGENT_TYPE == 'NOVELTOR':
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
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg, self.Qr, self.Qc], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)

        def select_action(self, t, greedy=False):
            if t.action != 'initialize':
                reward = 1/(self.Qs.visits[t.state][t.action] + 1) if t.state_ != t.state else 0
                self.Qe.update_RSS(t.action, reward, t.state_)
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

else:
    class ME_AIC_Learner:
        def __init__(self):
            if GRIDWORLD == 'WILLEMSEN':
                self.Qs = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.identity())
                self.Qg = MDP.RMax_table(alpha=ALPHA, gamma=GAMMA, mask=MDP.global_context())
                self.Ql = MDP.LinearEstimator(alpha=ALPHA, gamma=GAMMA)
                self.Qe = MDP.CombinedAIC([self.Qs, self.Qg, self.Ql], RSS_alpha=0.9, weights_method=WEIGHTS_METHOD)
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
    # print(transition)
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
    imV = np.reshape(V_vector, (1, env.grid_width)).T
    A_vector = [np.argmax(q) for q in Q_matrix]
    imA = np.reshape(A_vector, (1, env.grid_width)).T

    K = [e.count_parameters() for e in agent.Qe.estimators]
    RSS = agent.Qe.RSS

    print(agent.Ql.mx, agent.Ql.my)

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
plt.show()

# PLOT Estimator Evolution over time #

labels = [type(e.mask).__name__ for e in agent.Qe.estimators[:-1]] + ['linear']
W = []
for i in range(len(agent.Qe.estimators)):
    w = [ele['W'][i] for ele in metrics]
    W.append(w)
    plt.plot(w, label=labels[i])
plt.title('W over time')
plt.legend()
plt.show()

K = []
for i in range(len(agent.Qe.estimators)):
    k = [ele['K'][i] for ele in metrics]
    K.append(k)
    plt.plot(k, label=labels[i])
plt.title('K over time')
plt.legend()
plt.show()


for i in range(len(agent.Qe.estimators)):
    RSS = [ele['RSS'][i] for ele in metrics]
    plt.plot(RSS, label=labels[i])
plt.title('RSS over time')
plt.legend()
plt.show()


# Complexity PLOT
W = np.array(W)
K = np.array(K)
C = (W * K).sum(axis=0)  # matrix product
plt.title('Complexity over time')
plt.plot(C)
plt.show()


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


fig, ax = plt.subplots()
im = ax.imshow(metrics[-1]['V'], origin='lower')
ann_list = []
overlay_actions(metrics[-1]['A'])
plt.axis('off')
plt.show()
print(metrics[-1]['V'])
print(metrics[-1]['A'])


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
    writer = Writer(fps=15, metadata=dict(artist='Michael Kaisers'), bitrate=-1)
    ani.save(f"V_animation_{FILE_SIG}_-{VISUALISE_METRIC}.mp4", writer=writer)
