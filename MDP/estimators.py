from abc import ABC
from collections import defaultdict
import numpy as np


class Estimator:
    def __init__(self, approximator, mask, actions=None):
        if actions is None:
            self.actions = ['up', 'down', 'left', 'right']
        self.approximator = approximator
        self.mask = mask
        self.visits = defaultdict(lambda: {a: 0 for a in self.actions})

    def evaluate(self, transition):
        c = self.mask.apply(transition)
        return self.approximator.evaluate(c)

    def update(self, buffer_sample):
        for transition in buffer_sample:
            self.count_visits(transition)
            transition.N_ca = self.visits[transition.context][transition.action]
        self.approximator.update(buffer_sample)

    def count_visits(self, transition):
        context = self.mask.apply(transition)
        transition.context = context
        self.visits[transition.context][transition.action] += 1


class Mask(ABC):
    def __init__(self):
        pass

    def apply(self, transition):
        pass


class identity(Mask):
    def apply(self, transition):
        context = transition.state
        return context


class arrival_state(Mask):
    def apply(self, transition):
        context = transition.state_
        return context


class Approximator(ABC):
    def __init__(self):
        pass

    def update(self, buffer_sample):
        pass

    def evaluate(self, c, a):
        pass


class table(Approximator):
    def __init__(self, actions=None):
        super().__init__()
        if actions is None:
            actions = ['up', 'down', 'left', 'right']
        self.actions = actions
        self.table = defaultdict(lambda: {a: 0 for a in self.actions})

    def update(self, buffer_sample):
        for transition in buffer_sample:
            self.update_table(transition)
            self.update_value(transition)

    def update_table(self, transition):
        # if transition.state not in self.table.keys():
        #     self.table[transition.state] = {a: 0 for a in self.actions}
        # if transition.state_ not in self.table.keys():
        #     self.table[transition.state_] = {a: 0 for a in self.actions}
        pass

    def update_value(self, transition):
        self.table[transition.state][transition.action] += 1

    def evaluate(self, c):
        return self.table[c]


class state_table(table):
    def update_table(self, transition):
        if transition.state_ not in self.table.keys():
            self.table[transition.state_] = 0

    def update_value(self, transition):
        self.table[transition.state_] += 1


class bellman_Q_table(table):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def update_value(self, t):
        self.table[t.state][t.action] = self.table[t.state][t.action] + self.alpha * (t.reward + self.gamma * max(
            v for v in self.table[t.state_].values()) - self.table[t.state][t.action])


class bQt_novel_alpha(table):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def update_value(self, t):
        self.table[t.state][t.action] = self.table[t.state][t.action] + 1/t.N_ca * (t.reward + self.gamma * max(
            v for v in self.table[t.state_].values()) - self.table[t.state][t.action])


class bellman_N_table(bellman_Q_table):
    def update_value(self, t):
        self.table[t.state][t.action] = self.table[t.state][t.action] + self.alpha * (1/t.N_ca + self.gamma * max(
            v for v in self.table[t.state_].values()) - self.table[t.state][t.action])


class average(table):
    def update_value(self, t):
        self.table[t.context][t.action] = self.table[t.context][t.action] * (
                    t.N_ca - 1) / t.N_ca + t.reward * 1 / t.N_ca


class ema(table):
    def update_value(self, t):
        self.table[t.context][t.action] = self.table[t.context][t.action] * (1 - 0.25) + (1/t.N_ca) * 0.25


class global_N_abstractor(bellman_N_table):
    def evaluate(self, c):
        d = defaultdict(list)
        for s, actions in self.table.items():
            for a, value in actions.items():
                d[a].append(value)
        global_abstractor = {a: 0 for a in self.actions}
        for a, v in d.items():
            global_abstractor[a] = np.mean(v)
        return global_abstractor


class global_Q_abstractor(bellman_N_table):
    def evaluate(self, c):
        d = defaultdict(list)
        for s, actions in self.table.items():
            for a, value in actions.items():
                d[a].append(value)
        global_abstractor = {a: 0 for a in self.actions}
        for a, v in d.items():
            global_abstractor[a] = np.mean(v)
        return global_abstractor
