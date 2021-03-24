import numpy as np


class Agent:
    def __init__(self, estimators, buffer, targets, counter, batch_size=1):
        self.actions = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.actions)
        self.actions_map = {action: i for i, action in enumerate(self.actions)}
        self.trajectory = []
        self.batch_size = batch_size
        self.estimators = estimators
        self.buffer = buffer
        self.targets = targets
        self.counter = counter
        self.action_selection = None
        self.train = True

    def select_action(self, transition):
        self.process_state_estimators(transition)
        action = self.combine_state_estimators(transition)
        return action

    def observe(self, transition):
        if self.train:
            self.trajectory.append(transition)
            self.update_counter(transition)
            if transition.terminal:
                T = self.compute_trajectory_targets(self.trajectory)
                self.store_trajectory(self.trajectory, T)

    def process_state_estimators(self, transition):
        if self.train and self.buffer.size >= self.batch_size:
            S = self.buffer.sample(self.batch_size)
            for E in self.estimators:
                E.update(S)

    def combine_state_estimators(self, transition):
        action = self.actions[np.random.randint(self.n_actions)]
        return action

    def update_counter(self, transition):
        self.counter.update_table(transition)
        self.counter.update_value(transition)

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
    def combine_state_estimators(self, transition):
        # picks actions by maximising a random estimator
        maximizing_actions = []
        for E in self.estimators:
            action_values = E.evaluate(transition)
            max_value = max(action_values.values())
            max_actions = [k for k, v in action_values.items() if v == max_value]
            maximizing_actions.append(np.random.choice(max_actions))
        return np.random.choice(maximizing_actions)


class LambChop(Agent):
    def combine_state_estimators(self, transition):
        lam = np.zeros((self.n_actions, len(self.estimators)))
        Q = np.zeros((self.n_actions, len(self.estimators)))

        for a, A in enumerate(self.actions):
            n = np.zeros(len(self.estimators))
            b = np.zeros(len(self.estimators))
            Q_sa = self.estimators[0].evaluate(transition)[A]
            for e, E in enumerate(self.estimators):
                n[e] = E.get_visits(transition)[A]
                Q[a, e] = E.evaluate(transition)[A]
                b[e] = Q[a, e] - Q_sa

            # here where I check whether nb is 0 there may be a weird interaction with the initialized values
            # since when we calculate bias for an unseen state we initialize the value of the unseen q to 0
            nb = n + b ** 2
            Zigma = np.where(nb > 0, 1 / n, 0.1)  # Zigma is the Inverse of the Sigma MSE matrix
            den = np.sum(Zigma)
            lam[a] = Zigma / den

        lamQ = np.sum(lam * Q, axis=1)

        if self.action_selection == 'greedy':
            a = np.random.choice(np.argwhere(lamQ == np.max(lamQ)).flatten())
            action = self.actions[int(a)]

        elif self.action_selection == 'exploratory':
            exp_bonus = np.sqrt(np.sum((lam * Q) ** 2, axis=1))
            lamQ = lamQ + exp_bonus
            a = np.random.choice(np.argwhere(lamQ == np.max(lamQ)).flatten())
            action = self.actions[int(a)]

        return action


class QGreedyNoveltor(Agent):
    def combine_state_estimators(self, transition):
        # Epsilon Greedy Action Selection based on Q values
        if np.random.random() > 1 - self.epsilon:
            action = self.actions[np.random.randint(self.n_actions)]
        else:
            Qs = self.estimators[1].evaluate(transition)
            max_value = max(Qs.values())
            max_actions = [k for k, v in Qs.items() if v == max_value]
            action = np.random.choice(max_actions)
        return action


class GlobalNoveltor(Agent):
    def combine_state_estimators(self, transition):
        if self.action_selection == "novelty":
            Ns = self.estimators[1]
            if transition.state_ in Ns.approximator.table:
                # action selection based on state action values
                ns = Ns.evaluate(transition)
                max_value = max(ns.values())
                max_actions = [k for k, v in ns.items() if v == max_value]
                action = np.random.choice(max_actions)
            else:
                # action selection solely based on global abstractor
                ANg = self.estimators[2]
                ng = ANg.evaluate(transition)
                max_value = max(ng.values())
                max_actions = [k for k, v in ng.items() if v == max_value]
                action = np.random.choice(max_actions)
                # print(ng)
                # print(action)
        elif self.action_selection == "greedy":
            # Epsilon Greedy Action Selection based on Q values
            if np.random.random() > 1 - self.epsilon:
                action = self.actions[np.random.randint(self.n_actions)]
            else:
                Qs = self.estimators[0].evaluate(transition)
                max_value = max(Qs.values())
                max_actions = [k for k, v in Qs.items() if v == max_value]
                action = np.random.choice(max_actions)
                # print(Qs)
                # print(action)
        return action
