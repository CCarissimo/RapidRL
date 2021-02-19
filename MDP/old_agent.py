from buffer import ReplayMemory
import numpy as np


# Agent class, currently only selects random actions
class Agent:
    def __init__(self, mem_size=100000):
        self.actions = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.actions)
        self.actions_map = {action: i for i, action in enumerate(self.actions)}
        self.trajectory = []
        self.buffer = ReplayMemory(max_size=mem_size)
        self.targets = []
        self.estimators = []
        self.abstractors = []

    def select_action(self, state):
        self.process_state_estimators(state)
        action = self.combine_state_estimators(state)
        return action

    def update(self, transition):
        state, action, reward, state_, terminal = transition

        self.update_tables(transition)
        self.trajectory.append(transition)

        if terminal:
            T = self.compute_trajectory_targets(self.trajectory)
            self.store_trajectory(self.trajectory, T)

    def process_state_estimators(self, state):
        pass

    def combine_state_estimators(self, state):
        action = self.actions[np.random.randint(self.n_actions)]
        return action

    def update_tables(self, transition):
        pass

    def compute_trajectory_targets(self, trajectory):
        T = {}
        for target in self.targets:
            target.compute(trajectory)
        return T

    def store_trajectory(self, trajectory, targets):
        for transition in trajectory:
            transition.append(targets)
            self.buffer.append(transition)

    def reset_trajectory(self):
        self.trajectory = []


class Qlearner(Agent):
    def __init__(self, alpha, epsilon, gamma, batch_size, s_initial, s_prior=None, learn=True):
        super().__init__()

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.learn = learn

        self.trajectory = []

        initial_state_key = str(s_initial)
        self.visits = {initial_state_key: 1}
        if s_prior is None:
            self.q_table = {initial_state_key: {action: 0 for action in self.actions}}
        else:
            self.q_table = {initial_state_key: s_prior}

    def backup_q_value(self, s, a, s_, r):
        s = str(s)
        s_ = str(s_)
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (r + self.gamma * max(
            v for v in self.q_table[s_].values()) - self.q_table[s][a])

    def learn_from_past_experiences(self, ):
        transitions = self.buffer.sample(self.batch_size)

        for t in transitions:
            s, a, s_, r, _, targets = t
            self.backup_q_value(s, a, s_, r)

    def process_state_estimators(self, state):
        # sample a random batch of transitions, and use it to update the q_table
        if self.learn:
            if self.buffer.size > self.batch_size:
                self.learn_from_past_experiences()

    def combine_state_estimators(self, state):
        state_key = str(state)

        # select action that maximises q or explore a random action with probability epsilon
        rand = np.random.random()

        if rand > self.epsilon:
            action = max(self.q_table[state_key], key=self.q_table[state_key].get)
        else:
            action = np.random.choice(list(self.q_table[state_key].keys()))

        return action

    def update_tables(self, transition):
        state, action, state_, reward, terminal = transition
        state__key = str(state_)

        # check if state has been visited and update tables
        if state__key not in self.q_table.keys():
            self.q_table[state__key] = {action: 0 for action in self.actions}
            self.visits[state__key] = 0

        self.visits[state__key] += 1


class Abstractor(Qlearner):
    def __init__(self, alpha, epsilon, gamma, batch_size, s_initial, s_prior=None, learn=True):
        super().__init__(alpha, epsilon, gamma, batch_size, s_initial, s_prior=s_prior, learn=learn)
        self.action_abstractor = {action: 0 for action in self.actions}
        self.abstract = False

    def abstract_actions(self, ):
        N = {action: 0 for action in self.actions}
        for state, actions in self.q_table.items():
            for a, Q_sa in actions.items():
                self.action_abstractor[a] = self.action_abstractor[a] * N[a] / (N[a] + 1) + Q_sa / (N[a] + 1)
                N[a] += 1

    def combine_state_estimators(self, state):
        state_key = str(state)
        if self.abstract:
            self.abstract_actions()
            action = max(self.action_abstractor, key=self.action_abstractor.get)

        else:
            # select action that maximises q or explore a random action with probability epsilon
            rand = np.random.random()

            if rand > self.epsilon:
                action = max(self.q_table[state_key], key=self.q_table[state_key].get)
            else:
                action = np.random.choice(list(self.q_table[state_key].keys()))

        return action


# the main difference of the Noveltor is that he counts (state, action) pairs
# and updates a table of novelties as he updates the q_table
class Noveltor(Abstractor):
    def __init__(self, alpha, epsilon, gamma, batch_size, s_initial, s_prior=None, learn=True):
        super().__init__(alpha, epsilon, gamma, batch_size, s_initial, s_prior=s_prior, learn=learn)
        initial_state_key = str(s_initial)
        self.n_table = {initial_state_key: {a: 0 for a in self.actions}}
        self.n_visits = {initial_state_key: {action: 0 for action in self.actions}}

    def backup_q_value(self, s, a, s_, r):
        s = str(s)
        s_ = str(s_)
        alpha = 1/(self.n_visits[s][a])
        self.q_table[s][a] = self.q_table[s][a] + alpha * (r + self.gamma * max(
            v for v in self.q_table[s_].values()) - self.q_table[s][a])

    def backup_n_value(self, s, a, s_):
        s = str(s)
        s_ = str(s_)
        self.n_table[s][a] = self.n_table[s][a] + self.alpha * (1/(self.n_visits[s][a]) + self.gamma
                                                                * max(v for v in self.n_table[s_].values()) -
                                                                self.n_table[s][a])

    def learn_from_past_experiences(self):
        transitions = self.buffer.sample(self.batch_size)

        for t in transitions:
            s, a, s_, r, _, targets = t
            self.backup_q_value(s, a, s_, r)
            self.backup_n_value(s, a, s_)

    def update_tables(self, transition):
        state, action, state_, _, _ = transition

        # check if state has been visited and update tables
        state_key = str(state)
        state_key_ = str(state_)

        if state_key_ not in self.q_table.keys():
            self.q_table[state_key_] = {a: 0 for a in self.actions}
            self.n_table[state_key_] = {a: 0 for a in self.actions}
            self.n_visits[state_key_] = {a: 0 for a in self.actions}
            self.visits[state_key_] = 0
        self.n_visits[state_key][action] += 1
        self.visits[state_key_] += 1
