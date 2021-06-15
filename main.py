import numpy as np
from Agents import *
from Environments import *
from Utils import *
from warnings import filterwarnings
import matplotlib.pyplot as plt
import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("experiment", help="is it experiment A, B or C",
                    type=str)
parser.add_argument("gridworld", help="specify the name of the environment",
                    type=str)
parser.add_argument("agent_type", help="specify the agent type",
                    type=str)
parser.add_argument("--exploit", help="if included agent will alternate between exploration and exploitation to evaluate the learned information", action="store_false")
parser.add_argument("--plot", help="include if you would like plots", action="store_true")
parser.add_argument("--animate", help="include if you would like an animation of the value function"
    , action="store_true")
parser.add_argument("-n", "--max_steps", help="integer: the maximum number of steps, default at 100", type=int, default=100)
parser.add_argument("--timeout", help="integer: the maximum number of steps for a single trajectory, default at 34", type=int, default=34)
parser.add_argument("-a", "--alpha", help="float: the alpha value used for updates, default at 0.1", type=float, default=0.1)
parser.add_argument("--rss_alpha", help="float: the alpha value used for updates in calculating RSS, default at 0.1", type=float, default=0.1)
parser.add_argument("--lin_alpha", help="float: the alpha value used for updates in polynomial estimators, default at 0.0001", type=float, default=0.0001)
parser.add_argument("-g", "--gamma", help="float: the discount factor default at 0.9", type=float, default=0.9)
parser.add_argument("-B", "--batch_size", help="integer: the number of samples used in training batches, default at 10", type=int, default=10)
parser.add_argument("--weights_method", help="string: method used to calculate weights in estimator combination, default is exp_size_corrected", type=str, default="exp_size_corrected")
parser.add_argument("-e", "--epsilon", help="float: the exploration parameter for epsilon greedy exploration, default at 0.1", type=float, default=0.1)

args = parser.parse_args()

EXPERIMENT = args.experiment
GRIDWORLD = args.gridworld
AGENT_TYPE = args.agent_type
PLOT = True if args.plot else False
ANIMATE = True if args.animate else False
MAX_STEPS = args.max_steps
EPISODE_TIMEOUT = args.timeout
GAMMA = args.gamma
ALPHA = args.alpha
RSS_ALPHA = args.rss_alpha
LIN_ALPHA = args.lin_alpha
BATCH_SIZE = args.batch_size
WEIGHTS_METHOD = args.weights_method
EXPLOIT = args.exploit
EPSILON = args.epsilon

cwd = os.getcwd()
FOLDER = "%s\\Results" % (cwd)
FILE_SIG = f"{EXPERIMENT}_{AGENT_TYPE}_{GRIDWORLD}_n[{MAX_STEPS}]_alpha[{ALPHA}]_gamma[{GAMMA}]_batch[{BATCH_SIZE}]_weights[{WEIGHTS_METHOD}]_exploit[{EXPLOIT}]"
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
    # terminal_state.append([0, 8])
    # terminal_state.append([2, 8])
    terminal_state = np.array(terminal_state)
    initial_state = np.array([1, 0])
    blacked_state = np.array([[0, 8], [2, 8]])
elif GRIDWORLD == "STRAIGHT":
    grid = np.ones((3, 9)) * -1
    grid[1, :8] = 0
    grid[1, 8] = 1
    grid[2, :8] = 0.1
    terminal_state = []
    for i in [0]:
        for j in range(8):
            terminal_state.append([i, j])
    terminal_state.append([1, 8])
    # terminal_state.append([0, 8])
    # terminal_state.append([2, 8])
    terminal_state = np.array(terminal_state)
    initial_state = np.array([1, 0])
    blacked_state = np.array([[0, 8], [2, 8]])
elif GRIDWORLD == "POOL":
    grid = np.ones((8, 8)) * 0
    grid[3:6, 3:6] = -1
    grid[7, 7] = 1
    terminal_state = np.array([7, 7])
    initial_state = np.array([1, 0])
    blacked_state = np.array([[5, 5], [4, 5], [5, 4], [4, 4]])
    # blacked_state = np.array([[np.nan, np.nan], [np.nan, np.nan]])
elif GRIDWORLD == "STAR":
    grid = np.ones((15, 15)) * 0.1
    grid[7, 0] = -1
    grid[14, 7] = -1
    grid[7, 14] = 1
    grid[0, 7] = 1
    terminal_state = np.array([[7,0], [14,7], [7,14], [0,7]])
    initial_state = np.array([7,7])
    blacked_state = np.array([np.nan, np.nan])
# _, _, _, _ = plot_gridworld(grid, terminal_state, initial_state, blacked_state)

env = Gridworld(grid, terminal_state, initial_state, blacked_state, EPISODE_TIMEOUT)
env_greedy = Gridworld(grid, terminal_state, initial_state, blacked_state, EPISODE_TIMEOUT)

if GRIDWORLD == "WILLEMSEN":
    states = [(1, j) for j in range(env.grid_width)]
    env_shape = (1, env.grid_width)
elif GRIDWORLD == "STRAIGHT":
    states = [(i, j) for i in [0, 1, 2] for j in range(env.grid_width)]
    env_shape = (3, env.grid_width)
elif GRIDWORLD == "POOL" or GRIDWORLD == "STAR":
    states = [(i, j) for i in range(env.grid_height) for j in range(env.grid_width)]
    env_shape = (env.grid_height, env.grid_width)
else:
    print("WORLD not found")

print(states)

filterwarnings('ignore')

AGENTS = {
    "SIMPLE_Q": SimpleQ,
    "ME_Q": MEQ,
    "NOVELTOR": Noveltor,
    "RMAX": RMAXQ,
    }

agent = AGENTS[AGENT_TYPE]()
rb = ReplayMemory(max_size=10000)

trajectory = []
trajectories = []
metrics = []
epilen = []
Gn = []
step = 0

# MAIN TRAINING and EVALUATION LOOP
for i in tqdm.tqdm(range(MAX_STEPS)):
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
    if EXPLOIT:
        # RUN entire trajectory, and set greedy env to the initial state
        env_greedy.reset()
        while not env_greedy.terminal and not env_greedy.timeout:
            action = agent.select_action(env_greedy.transition, greedy=True)
            transition = env_greedy.step(action)
            Gg.append(transition.reward)

    step += 1

    RSS = agent.Qe.RSS
    AIC = agent.Qe.AIC

    # print(agent.Ql.W)
    K = [e.count_parameters() for e in agent.Qe.estimators]
    # Q_matrix = np.zeros((env.grid_height, env.grid_width, 4))
    Q_matrix = [agent.Qe.predict(s, store=False) for s in states]

    # for s in states:
    #     print(step, s, agent.Qe.prev_W, agent.Qe.predict(s, store=False))

    # print("Q", Q_matrix)
    V_vector = [np.max(q) for q in Q_matrix]
    # print("V", V_vector)
    imV = np.reshape(V_vector, (env_shape[0], env_shape[1]))
    A_vector = [np.argmax(q) for q in Q_matrix]  # randomize the max for ties
    imA = np.reshape(A_vector, (env_shape[0], env_shape[1]))

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
        'AIC': AIC,
        'steps': step
    })

    if env.terminal or env.timeout:
        trajectories.append(trajectory)
        epilen.append([len(Gn), len(Gg)])

        env.reset()
        trajectory = []
        Gn = []

# PLOT Cumulative Returns over time #
if PLOT:
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
    plt.xlabel('steps')

    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    ax2.set_ylabel('trajectory length')
    ax2.scatter(cumEpilen, [ele[0] for ele in epilen], s=5)
    ax2.scatter(cumEpiG, [ele[1] for ele in epilen], s=5)
    plt.tight_layout()
    plt.savefig(f'{FOLDER}\\G_{FILE_SIG}.png')

    # PLOT Estimator Evolution over time #

    labels = [type(e.mask).__name__ for e in agent.Qe.estimators]

    plt.figure(2)
    W = []
    for i in range(len(agent.Qe.estimators)):
        w = [ele['W'][i] for ele in metrics]
        W.append(w)
        plt.plot(w, label=labels[i])
    plt.title('W over time')
    plt.legend()
    plt.savefig(f'{FOLDER}\\W_{FILE_SIG}.png')

    plt.figure(3)
    K = []
    for i in range(len(agent.Qe.estimators)):
        k = [ele['K'][i] for ele in metrics]
        K.append(k)
        plt.plot(k, label=labels[i])
    plt.title('K over time')
    plt.legend()
    plt.savefig(f'{FOLDER}\\K_{FILE_SIG}.png')

    plt.figure(4)
    for i in range(len(agent.Qe.estimators)):
        RSS = [ele['RSS'][i] for ele in metrics]
        plt.plot(RSS, label=labels[i])
    plt.title('RSS over time')
    plt.legend()
    plt.savefig(f'{FOLDER}\\RSS_{FILE_SIG}.png')

    # Complexity PLOT
    plt.figure(5)
    W = np.array(W)
    K = np.array(K)
    C = (W * K).sum(axis=0)  # matrix product
    plt.title('Complexity over time')
    plt.plot(C)
    plt.savefig(f'{FOLDER}\\totK_{FILE_SIG}.png')

    # Value Function Plot

    visits = agent.Qs.visits
    hm = generate_heatmap(grid=env.grid, table=visits, aggf=lambda s: np.sum(s))
    plt.figure(6)
    plt.title(r"Updates Heatmap $\approx$ Visits")
    plt.imshow(hm, origin='lower')
    plt.savefig(f'{FOLDER}\\Visits_{FILE_SIG}.png')

    plt.figure(7)
    plt.title(r"AIC over time")
    for i in range(len(agent.Qe.estimators)):
        AIC = [ele['AIC'][i] for ele in metrics]
        plt.plot(AIC, label=labels[i])
    plt.legend()
    plt.savefig(f'{FOLDER}\\AIC_{FILE_SIG}.png')
    plt.show()

# ANIMATION

def overlay_actions(A):
    global ann_list
    k = 0
    for i in range(env.grid_width):
        for j in range(env.grid_height):
            if np.array([((j, i) == s).all() for s in env.terminal_states]).any():  # check my terminal states 
                terminal_reward = env.grid[j, i]
                if len(ann_list) > k:
                    ann_list[k].set_text(f"{terminal_reward:.1f}".lstrip('0'))
                    k += 1
                else:
                    text = ax.text(i, j, f"{terminal_reward:.1f}".lstrip('0'),
                                   ha="center", va="center", color="w")
                    ann_list.append(text)
                    k += 1
            else:
                if len(ann_list) > k:
                    ann_list[k].set_text(env.actions[A[j, i]][0:1])
                    k += 1
                else:
                    text = ax.text(i, j, env.actions[A[j, i]][0:1],
                                   ha="center", va="center", color="w")
                    ann_list.append(text)
                    k += 1


if ANIMATE:
    from matplotlib import animation, rc
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(env.grid_width, env.grid_height))
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad="3%")
    ax.axis('off')
    tx = ax.set_title('')
    # plt.tight_layout()
    fig.subplots_adjust(right=.8)
    im = ax.imshow(metrics[0]['V'], animated=True, origin='upper')
    # bars = ax2.bar(range(len(labels)), metrics[0]['W'], tick_label=labels, animated=True)
    # ax2.set_ylim(0, 1)
    ann_list = []

    def animate(i):
    # for i in range(len(metrics)):
        im.set_data(metrics[i]['V'])
        overlay_actions(metrics[i]['A'])
        cax.cla()
        cb = fig.colorbar(im, cax=cax)
        tx.set_text(f'V at {metrics[i]["steps"] / MAX_STEPS * 100:2.0f}% training')

        # for j, b in enumerate(bars):
        #     b.set_height(metrics[i]['W'][j])

    ani = animation.FuncAnimation(fig, animate, interval=40, frames=len(metrics))
    plt.show()

# Visualise Q-learning
# if ANIMATE:
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#     from matplotlib import animation, rc

#     # VISUALISE_METRIC = "N"
#     VISUALISE_METRIC = "V"
#     # VISUALISE_METRIC = "AD"

#     ann_list = []

#     fig = plt.figure(figsize=(env.grid_width, env.grid_height))
#     ax = fig.add_subplot(111)

#     div = make_axes_locatable(ax)
#     cax = div.append_axes("right", size="5%", pad="3%")

#     ax.axis('off')
#     ims = []

#     tx = ax.set_title('')
#     plt.tight_layout()
#     fig.subplots_adjust(right=.8)


#     def animate(i):
#         im = ax.imshow(metrics[i][VISUALISE_METRIC], animated=True, origin='lower')
#         if VISUALISE_METRIC == "V":
#             overlay_actions(metrics[i]['A'])
#         cax.cla()
#         cb = fig.colorbar(im, cax=cax)
#         tx.set_text(f'{VISUALISE_METRIC} at {metrics[i]["steps"] / MAX_STEPS * 100:2.0f}% training')


#     ani = animation.FuncAnimation(fig, animate, frames=len(metrics))

#     plt.close()

#     # HTML(ani.to_jshtml())
#     Writer = animation.writers['ffmpeg']
#     writer = Writer(fps=15, metadata=dict(artist='Michael Kaisers'), bitrate=100)
#     ani.save(f"V_animation_{FILE_SIG}_-{VISUALISE_METRIC}.mp4", writer=writer)  # extra_args=['-vcodec', 'libx264']
