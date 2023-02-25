from abc import ABC
from collections import defaultdict
import numpy as np


class Estimator(ABC):
    def __init__(self, mask, actions=None, initial_value=1):
        if actions is None:
            self.actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.mask = mask
        self.visits = defaultdict(lambda: np.zeros(len(self.actions)) + initial_value)
        if initial_value == "random":
            self.table = defaultdict(lambda: np.random.random(size=len(self.actions)))
        elif type(initial_value) is int:
            self.table = defaultdict(lambda: np.zeros(len(self.actions)) + initial_value)

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
    def __init__(self, mask, alpha, gamma, actions=None, initial_value=1):
        super().__init__(mask=mask, actions=actions, initial_value=1)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, t):
        c = self.mask.apply(t.state)
        c_ = self.mask.apply(t.state_)
        a = t.action
        nV = 0 if t.terminal else np.max(self.table[c_])
        # if nV > 0: print(nV)
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
    def update(self, t, novelty):
        c = self.mask.apply(t.state)
        c_ = self.mask.apply(t.state_)
        a = t.action
        self.visits[c][a] += 1

        if t.terminal:
            novelty = 0
            self.table[c_] = np.zeros(len(self.actions))
        elif t.state == t.state_:
            novelty = 0

        self.table[c][a] = self.table[c][a] + self.alpha * \
                       (novelty + self.gamma * max(self.table[c_]) - self.table[c][a])
        self.n += 1


class LinearEstimator:
    def __init__(self, alpha, gamma, mask, b=0, actions=None):
        if actions is None:
            self.actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.alpha = alpha
        self.gamma = gamma
        self.mask = mask  # expects the linear mask, which acts as identity, only for naming ... 
        self.n = 0
        self.mx = np.zeros(len(self.actions))
        self.my = np.zeros(len(self.actions))
        self.b = np.ones(len(self.actions)) * b
        self.W = np.array([self.mx, self.my, self.b]).T

    def update(self, t):
        """function to update the parameters of our linear estimator"""
        y, x = self.mask.apply(t.state)
        a = t.action
        y_hat = self.polynomial(x, y)[a]
        X = np.array([x, y, 1]).T
        y = t.reward + self.gamma * (np.abs(self.W[a, :-1]).max() + self.b[a])
        # print(self.W[a], X, y_hat, y)
        self.W[a] = self.W[a] - 2 * self.alpha * X * (y_hat - y)
        self.n += 1

    def polynomial(self, x, y):  # needs some notion of distance to compute the polynomial
        """takes the x and y positions on the grid to evaluate the value of a particular state"""
        return np.dot(np.array([x, y, 1]), self.W.T)

    def evaluate(self, s):  # returns an array of size len(actions)
        y, x = self.mask.apply(s)
        # print(self.polynomial(x, y))
        return self.polynomial(x, y)

    def get_visits(self, s):
        return self.n

    def count_parameters(self):
        return 13


class QuadraticEstimator:
    def __init__(self, alpha, gamma, mask, b=0, actions=None):
        if actions is None:
            self.actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.alpha = alpha
        self.gamma = gamma
        self.mask = mask  # expects the linear mask, which acts as identity, only for naming ... 
        self.n = 0
        self.mx = np.zeros(len(self.actions))
        self.my = np.zeros(len(self.actions))
        self.m2x = np.zeros(len(self.actions))
        self.m2y = np.zeros(len(self.actions))
        self.b = np.ones(len(self.actions)) * b
        self.W = np.array([self.mx, self.m2x, self.my, self.m2y, self.b]).T

    def update(self, t):
        """function to update the parameters of our linear estimator"""
        y, x = self.mask.apply(t.state)
        a = t.action
        y_hat = self.polynomial(x, y)[a]
        X = np.array([x, x ** 2, y, y ** 2, 1]).T
        y = t.reward + self.gamma * (max([np.abs(self.W[a, :2]).sum(), np.abs(self.W[a, :2]).sum()]) + self.b[a])
        # print(self.W[a], X, y_hat, y)
        self.W[a] = self.W[a] - 2 * self.alpha * X * (y_hat - y)
        self.n += 1

    def polynomial(self, x, y):  # needs some notion of distance to compute the polynomial
        """takes the x and y positions on the grid to evaluate the value of a particular state"""
        return np.dot(np.array([x, x ** 2, y, y ** 2, 1]), self.W.T)

    def evaluate(self, s):  # returns an array of size len(actions)
        y, x = self.mask.apply(s)
        # print(self.polynomial(x, y))
        return self.polynomial(x, y)

    def get_visits(self, s):
        return self.n

    def count_parameters(self):
        return 21


class LinearNoveltyEstimator(LinearEstimator):
    def __init__(self, alpha, gamma, mask, b=0, actions=None):
        super().__init__(alpha=alpha, gamma=gamma, mask=mask, b=b, actions=actions)
        self.visits = defaultdict(lambda: np.zeros(len(self.actions)))

    def update(self, t):
        """function to update the parameters of our linear estimator"""
        if t.terminal or t.state == t.state_:
            novelty = 0
        else:
            novelty = 1 / (self.visits[t.state][t.action] + 1)

        y, x = t.state
        y_hat = self.polynomial(x, y)
        y = novelty + self.gamma * (np.max(np.abs(self.W[:-1])) + self.b)
        X = np.array([x, y, 1]).T
        self.W = self.W - 2 * self.alpha * X * (y_hat - y)
        self.n += 1
        self.visits[t.state] += 1


class pseudoCountNovelty:
    def __init__(self, features: list, alpha: float):
        self.features = features
        self.tables = [defaultdict(lambda: float(0)) for mask in self.features]
        self.alpha = alpha
        self.t = 1

    def update(self, s):
        for i, mask in enumerate(self.features):
            c = mask.apply(s)
            self.tables[i][c] += 1
        self.t += 1

    def evaluate(self, s):
        """evaluation function for pseudo-counts by estimating feature occurrences"""
        C = [mask.apply(s) for mask in self.features]
        rho = np.array([self.tables[i][c] for i, c in enumerate(C)]) / self.t  # features before observation
        self.update(s)
        rho_ = np.array([self.tables[i][c] for i, c in enumerate(C)]) / self.t  # features after observation
        pseudoCount = rho.prod() * (1 - rho_.prod()) / (rho_.prod() - rho.prod())
        if pseudoCount == 0: pseudoCount += 0.0001
        return self.alpha / np.sqrt(pseudoCount)


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
    def __init__(self, estimators, RSS_alpha, beta=2, weights_method='exponential'):
        self.estimators = estimators
        self.prev_V = np.zeros((4, len(self.estimators)))  # 4 actions, a value for each action
        self.prev_W = np.zeros(len(self.estimators))
        self.RSS = np.ones(len(self.estimators)) * 1e-6
        self.alpha = RSS_alpha
        self.W = np.ones(len(self.estimators)) / len(self.estimators)
        self.weights_method = weights_method
        self.gamma = self.estimators[0].gamma
        self.beta = beta
        self.AIC = np.full([len(self.estimators)], np.nan)

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
        K = np.array([e.count_parameters() for e in self.estimators]).T
        # N = np.array([np.sum(e.get_visits(s)) for e in self.estimators]).T
        N = np.array([e.n for e in self.estimators]).T
        complexity = np.multiply(2, K)
        accuracy = np.multiply(N, np.log(self.RSS))
        if self.weights_method == "exponential":
            self.AIC = np.add(complexity, accuracy)
            aic = np.min(self.AIC)
            w = np.exp(np.subtract(aic, self.AIC) / self.beta)
        elif self.weights_method == "exp_size_corrected":
            self.AIC = np.add(np.add(complexity, accuracy),
                              np.add(2 * np.power(K, 2), 2 * K) / np.subtract(np.subtract(N, K), 1)) / np.where(N != 0,
                                                                                                                N, 1)
            aic = np.min(self.AIC)
            w = np.exp(np.subtract(aic, self.AIC) / self.beta)
        elif self.weights_method == "weighted_average":
            self.AIC = np.add(complexity, accuracy)
            w = 1 / self.AIC
        # baselines to compare our methods to
        # w_biased = sqrt(b/(n_u+b))
        # n_b/(n_u + n_b + 4*n_u*n_b*b)
        self.W = w / np.sum(w)
        # print('AIC Weights', K, N, complexity, accuracy, self.RSS, self.AIC, self.W)
        return self.W

    def predict(self, s, store=True):
        V = np.array([e.evaluate(s) for e in self.estimators]).T
        if store:
            self.prev_V = V
            self.prev_W = self.weights(s)
        # print('predict', V, self.prev_W)
        return np.dot(V, self.prev_W)  # Matrix @ Vector dot product

    def update_RSS(self, a, r, s_):  # un-discounted reward
        V = np.array([e.evaluate(s_) for e in self.estimators]).T
        # print("V", V.T, self.prev_V.T)
        Qsa = self.prev_V[a]  # check that this is a vector
        # print(self.prev_V)
        maxQs_a_ = np.max(np.dot(V, self.prev_W))  # check that this is a vector
        e = r + self.gamma * maxQs_a_ - Qsa
        e2 = np.power(e, 2)
        self.RSS = np.multiply((1 - self.alpha), self.RSS) + np.multiply(self.alpha, e2)
        # print("update_RSS:", V, Qsa, maxQs_a_, e, e2, self.RSS)
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
