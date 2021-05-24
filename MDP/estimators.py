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
        self.n = 0

    def update(self, t):
        context = self.mask.apply(t.state)
        self.visits[context][t.action] += 1
        self.n += 1

    def evaluate(self, s):
        c = self.mask.apply(s)
        return self.table[c]

    def get_visits(self, s):
        context = self.mask.apply(s)
        return self.visits[context]

    def UCB_bonus(self, s):
        return np.sqrt(np.log(1 + self.n) / (1 + self.visits[s]))  # * UCB Exploration COEFFICIENT

    def count_parameters(self):
        return len(self.table) * len(self.actions) + 1  # 1 residual for the estimator, more if local RSS


class Q_table(Estimator):
    def __init__(self, mask, alpha, gamma, actions=None):
        super().__init__(mask=mask, actions=actions)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, t):
        c = self.mask.apply(t.state)
        c_ = self.mask.apply(t.state_)
        a = t.action
        nV = 0 if t.terminal else max(self.table[c_])
        self.visits[c][a] += 1
        self.table[c][a] = self.table[c][a] + self.alpha * \
                           (t.reward + self.gamma * nV - self.table[c][a])
        self.n += 1


class RMax_table(Q_table):
    def __init__(self, mask, alpha, gamma, MAX=1, actions=None):
        super().__init__(mask, alpha, gamma, actions)
        self.MAX = MAX
        self.table = defaultdict(lambda: np.ones(len(self.actions)) * MAX)


class N_table(Q_table):
    def update(self, t):
        c = self.mask.apply(t.state)
        c_ = self.mask.apply(t.state_)
        a = t.action
        if t.terminal or t.state == t.state_:
            novelty = 0
        else:
            novelty = 1 / (self.visits[c][a]+1)

        self.visits[c][a] += 1
        self.table[c][a] = self.table[c][a] + self.alpha * \
                           (novelty + self.gamma * max(self.table[c_]) - self.table[c][a])
        self.n += 1


class CombinedActionEstimator:
    def __init__(self, estimators):
        self.estimators = estimators

    def weights(self, s):
        """ Computes weight matrix of size (n_actions, n_estimators) """
        N = np.array([e.get_visits(s) for e in self.estimators]).T
        MSE = 0.5 / np.sqrt(N + 1) + np.array([0, 0.2, 0.1, 0.1])  # BIAS
        W = 1 / MSE
        norm = np.abs(W).sum(axis=1)
        for i in range(len(norm)):
            if norm[i] == 0:
                W[i, :] = 1 / len(self.estimators)
        norm[norm == 0] = 1
        Wn = W / norm.reshape(len(norm), 1)
        return Wn

    def predict(self, s):
        V = np.array([e.evaluate(s) for e in self.estimators]).T
        return np.einsum('ij,ij->i', self.weights(s), V)  # row-wise dot product

    def UCB_bonus(self, s):
        V = np.array([e.UCB_bonus(s) for e in self.estimators]).T
        return np.einsum('ij,ij->i', self.weights(s), V)  # row-wise dot product

    def update(self, t):
        for e in self.estimators:
            e.update(t)
            # return np.mean(alphas)


class CombinedAIC:
    def __init__(self, estimators, RSS_alpha, weights_method='exponential'):
        self.estimators = estimators
        self.prev_V = np.zeros(len(self.estimators))
        self.prev_W = np.zeros(len(self.estimators))
        self.RSS = np.ones(len(self.estimators)) * 0.01
        self.alpha = RSS_alpha
        self.W = np.ones(len(self.estimators))/len(self.estimators)
        self.weights_method = weights_method

    # def weights(self, s):
    #     """Computes the weights for the estimators based on the Akaike Information Criterion"""
    #     K = np.array([e.count_parameters() for e in self.estimators]).T
    #     N = np.array([np.sum(e.get_visits(s)) for e in self.estimators]).T
    #     complexity = np.multiply(2, K)
    #     accuracy = np.multiply(N, np.log(self.RSS))
    #     AIC = np.subtract(complexity, accuracy)
    #     w = 1/AIC
    #     self.W = w/np.sum(w)
    #     # print('AIC Weights', K, N, complexity, accuracy, self.RSS, AIC, W)
    #     return self.W

    def weights(self, s):
        """Computes the weights for the estimators based on the Akaike Information Criterion"""
        if self.weights_method == "exponential":
            K = np.array([e.count_parameters() for e in self.estimators]).T
            N = np.array([np.sum(e.get_visits(s)) for e in self.estimators]).T
            complexity = np.multiply(2, K)
            accuracy = np.multiply(N, np.log(self.RSS))
            AIC = np.subtract(complexity, accuracy)
            aic = np.min(AIC)
            w = np.exp(np.subtract(aic, AIC)/2)
        elif self.weights_method == "weighted_average":
            K = np.array([e.count_parameters() for e in self.estimators]).T
            N = np.array([np.sum(e.get_visits(s)) for e in self.estimators]).T
            complexity = np.multiply(2, K)
            accuracy = np.multiply(N, np.log(self.RSS))
            AIC = np.subtract(complexity, accuracy)
            w = 1/AIC
        self.W = w/np.sum(w)
        # print('AIC Weights', K, N, complexity, accuracy, self.RSS, AIC, W)
        return self.W

    def predict(self, s):
        V = np.array([e.evaluate(s) for e in self.estimators]).T
        self.prev_V = V
        self.prev_W = self.weights(s)
        # print('predict', V, self.weights(s))
        return np.dot(V, self.prev_W)  # Matrix @ Vector dot product

    def update_RSS(self, a, r, s_): # un-discounted reward
        V = np.array([e.evaluate(s_) for e in self.estimators]).T
        gamma = self.estimators[0].gamma
        Qsa = np.dot(self.prev_V, self.prev_W)[a]
        maxQs_a_ = np.max(np.dot(V, self.prev_W))
        target = r + gamma * maxQs_a_ - Qsa
        e = np.subtract(np.multiply(r, target), self.prev_V[a])
        e2 = np.power(e, 2)
        # print('RSS', self.RSS, e, e2)
        self.RSS = np.multiply(self.alpha, self.RSS) + np.multiply((1-self.alpha), e2)
        # print('RSS2', self.RSS)
        return self.RSS

    def update(self, t):
        for e in self.estimators:
            e.update(t)

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
