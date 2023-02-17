# define training and evaluation loop 
from tqdm.auto import tqdm
import numpy as np


def train_and_eval(MAX_STEPS, BATCH_SIZE, EPISODE_TIMEOUT, agent, rb, env, env_greedy, states, env_shape, EXPLOIT=True):
    trajectory = []
    trajectories = []
    metrics = []
    epilen = []
    Gn = []
    step = 0

    # MAIN TRAINING and EVALUATION LOOP
    for i in tqdm(range(MAX_STEPS)):
        # EXPLORE
        action = agent.select_action(env.transition)
        # print('explore', action)
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
            while not env_greedy.terminal and len(Gg) <= EPISODE_TIMEOUT:
                action = agent.select_action(env_greedy.transition, greedy=True)
                # print('exploit', action)
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

        if env.terminal or len(trajectory) >= EPISODE_TIMEOUT:
            # trajectories.append(trajectory)
            epilen.append([len(Gn), len(Gg)])

            env.reset()
            trajectory = []
            Gn = []

    return metrics  # , trajectories
