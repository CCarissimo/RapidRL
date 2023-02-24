# define training and evaluation loop
import copy

from tqdm.auto import tqdm
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)


def online_learning(MAX_STEPS, BATCH_SIZE, EPISODE_TIMEOUT, agent, rb, env, states, env_shape):
    trajectory = []
    trajectories = []
    metrics = []
    trajectory_metrics = []
    epilen = []
    Gn = []
    step = 0
    transition = env.transition

    # MAIN TRAINING and EVALUATION LOOP
    for i in tqdm(range(MAX_STEPS)):
        # EXPLORE
        # pp.pprint(agent.Ns.table)
        action = agent.select_action(transition)
        # print('explore', action)
        transition = env.step(action)
        rb.append(transition)
        S = rb.sample(batch_size=BATCH_SIZE)
        agent.update(S)
        Gn.append(1 / (agent.visits[transition.state_] + 1))
        novelty = 1 / (agent.visits[transition.state_] + 1)
        agent.update_visits(transition)  # visits counting and novelty updates must be separated
        trajectory.append(transition)

        step += 1

        Q_matrix = [agent.Ns.evaluate(s) for s in states]

        V_vector = [np.max(q) for q in Q_matrix]

        imV = np.reshape(V_vector, (env_shape[0], env_shape[1]))
        A_vector = [np.argmax(q) for q in Q_matrix]  # randomize the max for ties
        imA = np.reshape(A_vector, (env_shape[0], env_shape[1]))

        metrics.append({
            't': i,
            'V': imV,
            'A': imA,
            'S': transition.state_,
            'Gn': Gn,
            'traj_len': len(Gn),
            "novelty": novelty,
            'steps': step
        })

        if env.terminal or len(trajectory) >= EPISODE_TIMEOUT:
            # trajectories.append(trajectory)

            trajectory_metrics.append({
                "visits": copy.deepcopy(agent.visits),
                "n_table": copy.deepcopy(agent.Ns.table),
                "lifetime": len(trajectory)
            })

            epilen.append(len(Gn))
            agent.reset_visits()
            rb.reset()
            env.reset()
            trajectory = []
            Gn = []

    return metrics, trajectory_metrics
