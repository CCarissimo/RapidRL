import numpy as np
from .estimators import *
from .masks import *


class MELearner:
    def __init__(self):
        self.Qs = Q_table(alpha=0.1, gamma=0.9, mask=identity())
        self.Qg = Q_table(alpha=0.1, gamma=0.9, mask=global_context())
        self.Qc = Q_table(alpha=0.1, gamma=0.9, mask=column())
        self.Qr = Q_table(alpha=0.1, gamma=0.9, mask=row())
        self.Qe = CombinedActionEstimator([self.Qs, self.Qg, self.Qc, self.Qr])

    def select_action(self, t, greedy=False):
        if greedy:
            values = self.Qe.predict(t.state)
            return np.random.choice(np.flatnonzero(values == values.max()))
        else:
            values = self.Qe.predict(t.state) + self.Qe.UCB_bonus(t.state)
            return np.random.choice(np.flatnonzero(values == values.max()))

    def update(self, transitions):
        for t in transitions:
            self.Qe.update(t)


class Agent:
    def __init__(self, estimators, buffer, targets, batch_size=1):
        self.actions = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.actions)
        self.actions_map = {action: i for i, action in enumerate(self.actions)}
        self.trajectory = []
        self.batch_size = batch_size
        self.estimators = estimators
        self.buffer = buffer
        self.targets = targets
        self.action_selection = None

    def select_action(self, transition):
        action = self.actions[np.random.randint(self.n_actions)]
        return action

    def observe(self, transition):
        self.trajectory.append(transition)
        self.buffer.append(transition)

    def train(self):
        if self.buffer.size >= self.batch_size:
            S = self.buffer.sample(self.batch_size)
            for transition in S:
                for E in self.estimators:
                    E.update(transition)

    def compute_trajectory_targets(self, trajectory):
        for target in self.targets:
            target.compute(trajectory)
        return 0

    def store_trajectory(self, trajectory, targets):
        for transition in trajectory:
            transition.targets = targets
            self.buffer.append(transition)

    def reset_trajectory(self):
        self.trajectory = []


class Greedy(Agent):
    def select_action(self, transition):
        # picks actions by maximising a random estimator
        maximizing_actions = []
        for E in self.estimators:
            action_values = E.evaluate(transition)
            max_value = np.max(action_values)
            max_actions = np.argwhere(action_values == max_value)
            maximizing_actions.append(np.random.choice(max_actions))
        return np.random.choice(maximizing_actions)


class eGreedy(Agent):
    def __init__(self, estimators, buffer, targets, epsilon, batch_size=1):
        super().__init__(estimators, buffer, targets, batch_size=1)
        self.epsilon = epsilon

    def select_action(self, transition):
        # assumes the action value estimator is the 0th estimator
        Q_s = self.estimators[0].evaluate(transition) # array of action values in state s

        if self.action_selection == 'exploratory':
            if np.random.random() > 1 - self.epsilon:
                action = self.actions[np.random.randint(self.n_actions)]
            else:
                max_value = np.max(Q_s)
                max_actions = np.argwhere(Q_s == max_value)
                action = self.actions[np.random.choice(max_actions)]

        elif self.action_selection == 'greedy':
            max_value = np.max(Q_s)
            max_actions = np.argwhere(Q_s == max_value)
            print(max_actions)
            action = self.actions[np.random.choice(max_actions)]

        return action


class LambChop(Agent):
    def select_action(self, transition):
        lam = np.zeros((self.n_actions, len(self.estimators)))
        Q = np.zeros((self.n_actions, len(self.estimators)))

        for a, A in enumerate(self.actions):
            n = np.zeros(len(self.estimators))
            b = np.zeros(len(self.estimators))
            Q_sa = self.estimators[0].evaluate(transition)[a]
            for e, E in enumerate(self.estimators):
                n[e] = E.get_visits(transition)[a]
                Q[a, e] = E.evaluate(transition)[a]
                b[e] = Q[a, e] - Q_sa

            # here where I check whether nb is 0 there may be a weird interaction with the initialized values
            # since when we calculate bias for an unseen state we initialize the value of the unseen q to 0
            Sigma = n + b ** 2 # this is just the diagonals of the MSE matrix
            Zigma = np.where(Sigma > 0, 1 / Sigma, 0.1)  # Zigma is the Inverse of the Sigma MSE matrix
            den = np.sum(Zigma)  # can not remember why set Zigma to 0.1 when Sigma <= 0
            lam[a] = Zigma / den

        lamQ = np.sum(lam * Q, axis=1)

        # check that lam sums to 1
        # print(np.sum(lam, axis=1))

        if self.action_selection == 'greedy':
            a = np.random.choice(np.argwhere(lamQ == np.max(lamQ)).flatten())
            action = self.actions[int(a)]

        elif self.action_selection == 'exploratory':
            t = sum(self.counter.table.values())
            n_s = self.estimators[0].get_visits(transition)  # maybe calculate earlier?
            exp_bonus = np.where(n_s > 0, np.sqrt(np.log(t)/n_s), 1000)
            # print(exp_bonus)
            lamQ = lamQ + exp_bonus
            # print(lamQ)
            a = np.random.choice(np.argwhere(lamQ == np.max(lamQ)).flatten())
            action = self.actions[int(a)]

        return action


class QGreedyNoveltor(eGreedy):
    def select_action(self, transition):
        # Epsilon Greedy Action Selection based on Q values
        if np.random.random() > 1 - self.epsilon:
            action = self.actions[np.random.randint(self.n_actions)]
        else:
            Qs = self.estimators[1].evaluate(transition)
            max_value = max(Qs.values())
            max_actions = np.argwhere(Qs == max_value)
            action = self.actions[np.random.choice(max_actions)]

        return action


class GlobalNoveltor(Agent):
    def select_action(self, transition):
        if self.action_selection == "novelty":
            visits = np.sum(self.estimators[1].get_visits())
            if visits > 0:
                # action selection based on state action values
                N_s = self.estimators[1].evaluate(transition)
                max_value = np.max(N_s)
                max_actions = np.argwhere(N_s == max_value)
                action = self.actions[np.random.choice(max_actions)]
            else:
                # action selection solely based on global abstractor
                N_g = self.estimators[2].evaluate(transition)
                max_value = np.max(N_g)
                max_actions = np.argwhere(N_g == max_value)
                action = self.actions[np.random.choice(max_actions)]
                # print(ng)
                # print(action)
        elif self.action_selection == "greedy":
            # Epsilon Greedy Action Selection based on Q values
            if np.random.random() > 1 - self.epsilon:
                action = self.actions[np.random.randint(self.n_actions)]
            else:
                Q_s = self.estimators[0].evaluate(transition)
                max_value = np.max(Q_s)
                max_actions = np.argwhere(Q_s == max_value)
                action = self.actions[np.random.choice(max_actions)]
                # print(Qs)
                # print(action)
        return action
