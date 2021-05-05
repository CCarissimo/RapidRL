from abc import ABC
from collections import defaultdict
import numpy as np


class Estimator(ABC):
    def __init__(self, mask, actions=None):
        if actions is None:
            self.actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.mask = mask
        self.visits = defaultdict(lambda: np.zeros(len(self.actions)))
        self.table = defaultdict(lambda: np.zeros(len(self.actions)))

    def update(self, buffer_sample):
        for transition in buffer_sample:
            context = self.mask.apply(transition.state)
            action = self.actions[transition.action]
            self.visits[context][action] += 1

    def evaluate(self, transition):
        c = self.mask.apply(transition.state)
        return self.table[c]

    def get_visits(self, transition):
        context = self.mask.apply(transition.state)
        return self.visits[context]


class Q_table(Estimator):
    def __init__(self, mask, alpha, gamma, actions=None):
        super().__init__(mask=mask, actions=actions)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, buffer_sample):
        for t in buffer_sample:
            c = self.mask.apply(t.state)
            c_ = self.mask.apply(t.state_)
            a = self.actions[t.action]
            self.visits[c][a] += 1
            self.table[c][a] = self.table[c][a] + self.alpha * \
                               (t.reward + self.gamma * max(self.table[c_]) - self.table[c][a])


class RMax_table(Q_table):
    def __init__(self, mask, alpha, gamma, MAX=1, actions=None):
        super().__init__(mask, alpha, gamma, actions)
        self.MAX = MAX
        self.table = defaultdict(lambda: {a: MAX for a in self.actions})


class N_table(Q_table):
    def update(self, buffer_sample):
        for t in buffer_sample:
            c = self.mask.apply(t.state)
            c_ = self.mask.apply(t.state_)
            a = self.actions[t.action]
            if t.terminal or t.state == t.state_:
                novelty = 0
            else:
                novelty = 1 / self.visits[c][a]

            self.visits[c][a] += 1
            self.table[c][a] = self.table[c][a] + self.alpha * \
                               (novelty + self.gamma * max(self.table[c_]) - self.table[c][a])


# class Estimator:
#     def __init__(self, mask, actions=None):
#         if actions is None:
#             self.actions = ['up', 'down', 'left', 'right']
#         self.mask = mask
#         self.visits = defaultdict(lambda: {a: 0 for a in self.actions})
#         self.table = defaultdict(lambda: {a: 0 for a in self.actions})
#
#     def evaluate(self, transition):
#         c = self.mask.apply(transition)
#         return self.table[c]
#
#     def update(self, buffer_sample):
#         for transition in buffer_sample:
#             self.count_visits(transition)
#             N_ca = self.visits[transition.context][transition.action]
#         self.table[transition.state][transition.action] += 1
#
#         self.approximator.update(buffer_sample)
#
#     def get_visits(self, transition):
#         context = self.mask.apply(transition)
#         transition.context = context
#         return self.visits[transition.context]
#
#     def count_visits(self, transition):
#         context = self.mask.apply(transition)
#         transition.context = context
#         self.visits[transition.context][transition.action] += 1


# class Approximator(ABC):
#     def __init__(self):
#         pass
#
#     def update(self, buffer_sample):
#         pass
#
#     def evaluate(self, c, a):
#         pass
#
#
# class table(Approximator):
#     def __init__(self, actions=None):
#         super().__init__()
#         if actions is None:
#             actions = ['up', 'down', 'left', 'right']
#         self.actions = actions
#         self.table = defaultdict(lambda: {a: 0 for a in self.actions})
#
#     def update(self, buffer_sample):
#         for transition in buffer_sample:
#             self.update_table(transition)
#             self.update_value(transition)
#
#     def update_table(self, transition):
#         # if transition.state not in self.table.keys():
#         #     self.table[transition.state] = {a: 0 for a in self.actions}
#         # if transition.state_ not in self.table.keys():
#         #     self.table[transition.state_] = {a: 0 for a in self.actions}
#         pass
#
#     def update_value(self, transition):
#         self.table[transition.state][transition.action] += 1
#
#     def evaluate(self, c):
#         return self.table[c]
#
#
# class state_table(table):
#     def update_table(self, transition):
#         if transition.state_ not in self.table.keys():
#             self.table[transition.state_] = 0
#
#     def update_value(self, transition):
#         self.table[transition.state_] += 1


# class bellman_Q_table(table):
#     def __init__(self, alpha, gamma):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def update_value(self, t):
#         self.table[t.state][t.action] = self.table[t.state][t.action] + self.alpha * (t.reward + self.gamma * max(
#             v for v in self.table[t.state_].values()) - self.table[t.state][t.action])


# class bellman_RMax(bellman_Q_table):
#     def __init__(self, alpha, gamma):
#         super().__init__(alpha=alpha, gamma=gamma)
#
#     def update_value(self, t):
#         self.table[t.state][t.action] = self.table[t.state][t.action] + self.alpha * (t.reward + self.gamma * max(
#             v for v in self.table[t.state_].values()) - self.table[t.state][t.action])


# class Q_ga_RMax(bellman_RMax):
#     def evaluate(self, c):
#         d = defaultdict(list)
#         for s, actions in self.table.items():
#             for a, value in actions.items():
#                 d[a].append(value)
#         global_abstractor = {a: 0 for a in self.actions}
#         for a, v in d.items():
#             global_abstractor[a] = np.mean(v)
#         return global_abstractor


# class bQt_novel_alpha(table):
#     def __init__(self, gamma):
#         super().__init__()
#         self.gamma = gamma
#
#     def update_value(self, t):
#         self.table[t.state][t.action] = self.table[t.state][t.action] + 1/t.N_ca * (t.reward + self.gamma * max(
#             v for v in self.table[t.state_].values()) - self.table[t.state][t.action])


# class bellman_N_table(bellman_Q_table):
#     def __init__(self, alpha, gamma):
#         super().__init__(alpha=alpha, gamma=gamma)
#         self.table = defaultdict(lambda: {a: 1 for a in self.actions})
#
#     def update_table(self, t):
#         if t.terminal:
#             self.table[t.state_] = {a: 0 for a in self.actions}
#
#     def update_value(self, t):
#         if t.terminal:
#             novelty_reward = 0
#         elif t.state == t.state_:
#             novelty_reward = 0
#         else:
#             novelty_reward = 1/t.N_ca
#
#         self.table[t.state][t.action] = self.table[t.state][t.action] + self.alpha * (novelty_reward + self.gamma * max(
#             v for v in self.table[t.state_].values()) - self.table[t.state][t.action])


# class average(table):
#     def update_value(self, t):
#         self.table[t.context][t.action] = self.table[t.context][t.action] * (
#                     t.N_ca - 1) / t.N_ca + t.reward * 1 / t.N_ca
#
#
# class ema(table):
#     def update_value(self, t):
#         self.table[t.context][t.action] = self.table[t.context][t.action] * (1 - 0.25) + (1/t.N_ca) * 0.25


# class global_N_abstractor(bellman_N_table):
#     def evaluate(self, c):
#         d = defaultdict(list)
#         for s, actions in self.table.items():
#             for a, value in actions.items():
#                 d[a].append(value)
#         global_abstractor = {a: 0 for a in self.actions}
#         for a, v in d.items():
#             global_abstractor[a] = np.mean(v)
#         return global_abstractor
#
#
# class global_Q_abstractor(bQt_novel_alpha):
#     def evaluate(self, c):
#         d = defaultdict(list)
#         for s, actions in self.table.items():
#             for a, value in actions.items():
#                 d[a].append(value)
#         global_abstractor = {a: 0 for a in self.actions}
#         for a, v in d.items():
#             global_abstractor[a] = np.mean(v)
#         return global_abstractor
#
#
# class Q_ga(bellman_Q_table):
#     def evaluate(self, c):
#         d = defaultdict(list)
#         for s, actions in self.table.items():
#             for a, value in actions.items():
#                 d[a].append(value)
#         global_abstractor = {a: 0 for a in self.actions}
#         for a, v in d.items():
#             global_abstractor[a] = np.mean(v)
#         return global_abstractor
