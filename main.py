import numpy as np
from Agents import *
from Environments import *
from Experiments import *
from Utils import *
from warnings import filterwarnings
import matplotlib.pyplot as plt
import tqdm
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("experiment", help="is it experiment A, B or C",
                    type=str)
parser.add_argument("gridworld", help="specify the name of the environment",
                    type=str)
parser.add_argument("agent_type", help="specify the agent type",
                    type=str)
parser.add_argument("--noexploit", help="if included agent will alternate between exploration and exploitation to evaluate the learned information", action="store_false")
parser.add_argument("--plot", help="include if you would like plots", action="store_true")
parser.add_argument("--animate", help="include if you would like an animation of the value function"
    , action="store_true")
parser.add_argument("--plotGW", help="include if you would like a plot of the Gridworld before training and evaluation is run", action="store_true")
parser.add_argument("-n", "--max_steps", help="integer: the maximum number of steps, default at 100", type=int, default=100)
parser.add_argument("--timeout", help="integer: the maximum number of steps for a single trajectory, default at 34", type=int, default=34)
parser.add_argument("-a", "--alpha", help="float: the alpha value used for updates, default at 0.1", type=float, default=0.1)
parser.add_argument("--rss_alpha", help="float: the alpha value used for updates in calculating RSS, default at 0.1", type=float, default=0.1)
parser.add_argument("--lin_alpha", help="float: the alpha value used for updates in polynomial estimators, default at 0.0001", type=float, default=0.0001)
parser.add_argument("-g", "--gamma", help="float: the discount factor default at 0.9", type=float, default=0.9)
parser.add_argument("-B", "--batch_size", help="integer: the number of samples used in training batches, default at 10", type=int, default=10)
parser.add_argument("--buffer_size", help="integer: the number of samples stored in the replay buffer", type=int, default=10000)
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
BUFFER_SIZE = args.buffer_size
RSS_ALPHA = args.rss_alpha
LIN_ALPHA = args.lin_alpha
BATCH_SIZE = args.batch_size
WEIGHTS_METHOD = args.weights_method
EXPLOIT = args.noexploit
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
            terminal_state.append((i, j))
    terminal_state.append((1, 8))
    terminal_reward = set(terminal_state)
    initial_state = (1, 0)
    blacked_state = {(0, 8), (2, 8)}

elif GRIDWORLD == "STRAIGHT":
    grid = np.ones((3, 9)) * -1
    grid[1, :8] = 0
    grid[1, 8] = 1
    grid[2, :8] = 0.1
    terminal_state = []
    for i in [0]:
        for j in range(8):
            terminal_state.append((i, j))
    terminal_state.append((1, 8))
    terminal_state = set(terminal_state)
    initial_state = (1, 0)
    blacked_state = {(0, 8), (2, 8)}

elif GRIDWORLD == "POOL":
    grid = np.ones((8, 8)) * 0
    grid[3:6, 3:6] = -1
    grid[7, 7] = 1
    terminal_state = {(7, 7)}
    initial_state = (1, 0)
    blacked_state = {(5, 5), (4, 5), (5, 4), (4, 4)}

elif GRIDWORLD == "STAR":
    grid = np.ones((15, 15)) * 0.1
    grid[7, 0] = -1
    grid[14, 7] = -1
    grid[7, 14] = 1
    grid[0, 7] = 1
    terminal_state = {(7,0), (14,7), (7,14), (0,7)}
    initial_state = (7,7)
    blacked_state = {(np.nan, np.nan)}

elif GRIDWORLD == "DEATH":
    world_size = 10
    grid = np.zeros((world_size, world_size))
    np.random.seed(seed=0)
    x_random_death_states = np.random.randint(1, world_size, size=world_size)
    np.random.seed(seed=1)
    y_random_death_states = np.random.randint(1, world_size, size=world_size)
    terminal_state = set(zip(y_random_death_states, x_random_death_states))
    initial_state = (0, 0)
    blacked_state = {}

if args.plotGW:
    _, _, _, _ = plot_gridworld(grid, terminal_state, initial_state, blacked_state)

env = Gridworld(grid, terminal_state, initial_state, blacked_state)
env_greedy = Gridworld(grid, terminal_state, initial_state, blacked_state)

if GRIDWORLD == "STRAIGHT" or GRIDWORLD=="WILLEMSEN":
    states = [(i, j) for i in [0, 1, 2] for j in range(env.grid_width)]
    env_shape = (3, env.grid_width)
elif GRIDWORLD == "POOL" or GRIDWORLD == "STAR" or GRIDWORLD == "DEATH":
    states = [(i, j) for i in range(env.grid_height) for j in range(env.grid_width)]
    env_shape = (env.grid_height, env.grid_width)
else:
    print("WORLD not found")

filterwarnings('ignore')

AGENTS = {
    "SIMPLE_Q": SimpleQ,
    "ME_Q": MEQ,
    "SIMPLE_NOVELTOR": SimpleNoveltor,
    "NOVELTOR": Noveltor,
    "RMAX": RMAXQ,
    "PSEUDOCOUNT": Pseudocount
    }

agent = AGENTS[AGENT_TYPE](ALPHA=ALPHA, GAMMA=GAMMA, initial_value="random")
rb = ReplayMemory(max_size=BUFFER_SIZE, len_death_memories=2)

trajectory = []
trajectories = []
metrics = []
epilen = []
Gn = []
step = 0

# MAIN TRAINING and EVALUATION LOOP

metrics, trajectory_metrics = online_learning(MAX_STEPS, BATCH_SIZE, EPISODE_TIMEOUT, agent, rb, env,
                                              states, env_shape)

with open("n_values", "wb") as file:
    pickle.dump([metrics[i]['V'] for i in range(len(metrics))], file)

# PLOT Cumulative Returns over time #
if PLOT:
    x = np.arange(0, metrics[-1]['steps'])
    yn = []
    # yg = []
    for i, metric in enumerate(metrics):
        Yn = 0
        # Yg = 0
        for j, g in enumerate(metric['Gn']):
            Yn += GAMMA ** j * g
        # for j, g in enumerate(metric['Gg']):
        #     Yg += GAMMA ** j * g

        yn.append(Yn)
        # yg.append(Yg)

    cumEpilen = np.cumsum([ele[0] for ele in epilen])
    cumEpiG = np.cumsum([ele[1] for ele in epilen])

    # plt.figure(0)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 5))
    ax1.plot(x, yn, label='exploring')

    # ax1.plot(x, yg, label='exploiting')

    ax1.set_ylabel('cumulative discounted reward')
    plt.xlabel('steps')

    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    ax2.set_ylabel('trajectory length')
    plt.plot([metrics[t]["traj_len"] for t in range(MAX_STEPS)])
    # ax2.scatter(cumEpilen, [ele[0] for ele in epilen], s=5)
    # ax2.scatter(cumEpiG, [ele[1] for ele in epilen], s=5)
    plt.tight_layout()
    plt.savefig(f'{FOLDER}\\G_{FILE_SIG}.png')

    plt.figure(0)
    plt.plot([metrics[t]["novelty"] for t in range(MAX_STEPS)])
    plt.ylabel("Found Novelty")
    plt.xlabel("timestep")

    # PLOT Estimator Evolution over time #

    # labels = [type(e.mask).__name__ for e in agent.Qe.estimators]
    #
    # plt.figure(2)
    # W = []
    # for i in range(len(agent.Qe.estimators)):
    #     w = [ele['W'][i] for ele in metrics]
    #     W.append(w)
    #     plt.plot(w, label=labels[i])
    # plt.title('W over time')
    # plt.legend()
    # plt.savefig(f'{FOLDER}\\W_{FILE_SIG}.png')
    #
    # plt.figure(3)
    # K = []
    # for i in range(len(agent.Qe.estimators)):
    #     k = [ele['K'][i] for ele in metrics]
    #     K.append(k)
    #     plt.plot(k, label=labels[i])
    # plt.title('K over time')
    # plt.legend()
    # plt.savefig(f'{FOLDER}\\K_{FILE_SIG}.png')
    #
    # plt.figure(4)
    # for i in range(len(agent.Qe.estimators)):
    #     RSS = [ele['RSS'][i] for ele in metrics]
    #     plt.plot(RSS, label=labels[i])
    # plt.title('RSS over time')
    # plt.legend()
    # plt.savefig(f'{FOLDER}\\RSS_{FILE_SIG}.png')
    #
    # # Complexity PLOT
    # plt.figure(5)
    # W = np.array(W)
    # K = np.array(K)
    # C = (W * K).sum(axis=0)  # matrix product
    # plt.title('Complexity over time')
    # plt.plot(C)
    # plt.savefig(f'{FOLDER}\\totK_{FILE_SIG}.png')

    # Value Function Plot

    visits = agent.visits
    hm = generate_heatmap(grid=env.grid, table=visits, aggf=lambda s: np.sum(s))
    plt.figure(6)
    plt.title(r"Updates Heatmap $\approx$ Visits")
    plt.imshow(hm, origin='lower')
    plt.savefig(f'{FOLDER}\\Visits_{FILE_SIG}.png')

    # plt.figure(7)
    # plt.title(r"AIC over time")
    # for i in range(len(agent.Qe.estimators)):
    #     AIC = [ele['AIC'][i] for ele in metrics]
    #     plt.plot(AIC, label=labels[i])
    # plt.legend()
    # plt.savefig(f'{FOLDER}\\AIC_{FILE_SIG}.png')
    
    if not ANIMATE: plt.show()


# ANIMATION
def overlay_actions(A):
    global ann_list
    k = 0
    for i in range(env.grid_width):
        for j in range(env.grid_height):
            if (j, i) in env.terminal_states:  # check my terminal states
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

    fig = plt.figure(figsize=(env.grid_width, env.grid_height))
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad="3%")
    ax.axis('off')
    tx = ax.set_title('')
    # plt.tight_layout()
    fig.subplots_adjust(right=.8)
    im = ax.imshow(metrics[0]['V'], animated=True, origin='upper', vmin=0, vmax=1/(1-GAMMA))
    cb = fig.colorbar(im, cax=cax)
    # bars = ax2.bar(range(len(labels)), metrics[0]['W'], tick_label=labels, animated=True)
    # ax2.set_ylim(0, 1)
    ann_list = []

    def animate(i):
    # for i in range(len(metrics)):
        arr = metrics[i]['V']
        vmax = np.max(arr)
        vmin = 0  # np.min(arr)
        
        im.set_data(arr)
        im.set_clim(vmin, vmax)
        overlay_actions(metrics[i]['A'])

        tx.set_text(f'V at {metrics[i]["steps"] / MAX_STEPS * 100:2.0f}% training')

        # for j, b in enumerate(bars):
        #     b.set_height(metrics[i]['W'][j])

    ani = animation.FuncAnimation(fig, animate, interval=int(1), frames=len(metrics))
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
