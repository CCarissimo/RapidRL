import numpy as np
from .estimators import *
from .masks import *

class Noveltor:
    def __init__(self, dim2D=False, ALPHA=0.1, GAMMA=0.9, RSS_alpha=0.1, LIN_alpha=0.0001, WEIGHTS_METHOD='exp_size_corrected'):
        self.Qs = Q_table(alpha=ALPHA, gamma=GAMMA, mask=identity())
        if dim2D:
            self.Ns = N_table(alpha=ALPHA, gamma=GAMMA, mask=identity())
            self.Ng = N_table(alpha=ALPHA, gamma=GAMMA, mask=global_context())
            self.Qe = CombinedAIC([self.Ns, self.Ng], RSS_alpha=ALPHA, weights_method=WEIGHTS_METHOD)
        else:
            self.Ns = N_table(alpha=ALPHA, gamma=GAMMA, mask=identity())
            self.Ng = N_table(alpha=ALPHA, gamma=GAMMA, mask=global_context())
            self.Nc = N_table(alpha=ALPHA, gamma=GAMMA, mask=column())
            self.Nr = N_table(alpha=ALPHA, gamma=GAMMA, mask=row())
            # self.Nl = LinearNoveltyEstimator(alpha=ALPHA, gamma=GAMMA)
            self.Qe = CombinedAIC([self.Ns, self.Ng, self.Nr, self.Nc], RSS_alpha=ALPHA, weights_method=WEIGHTS_METHOD)

    def select_action(self, t, greedy=False):
        if t.action != 'initialize' and greedy:
            reward = 1/(self.Ns.visits[t.state][t.action] + 1) if t.state_ != t.state else 0
            self.Qe.update_RSS(t.action, reward, t.state_)
        if greedy:  # use estimator with minimum RSS
            values = self.Qs.table[t.state]
            # print(t, values)
            return np.random.choice(np.flatnonzero(values == values.max()))
        else:  # use the AIC combined estimators
            values = self.Qe.predict(t.state)
            return np.random.choice(np.flatnonzero(values == values.max()))

    def update(self, transitions):
        for t in transitions:
            self.Qs.update(t)
            self.Qe.update(t)

class LinearNoveltor:
    def __init__(self, dim2D=False, ALPHA=0.1, GAMMA=0.9, RSS_alpha=0.1, LIN_alpha=0.0001, WEIGHTS_METHOD='exp_size_corrected'):
        if dim2D:
            self.Qs = N_table(alpha=ALPHA, gamma=GAMMA, mask=identity())
            self.Qg = N_table(alpha=ALPHA, gamma=GAMMA, mask=global_context())
            self.Ql = LinearNoveltyEstimator(alpha=ALPHA, gamma=GAMMA, mask=linear())
            self.Qe = CombinedAIC([self.Qs, self.Qg, self.Ql], RSS_alpha=ALPHA, weights_method=WEIGHTS_METHOD)
        else:
            self.Qs = N_table(alpha=ALPHA, gamma=GAMMA, mask=identity())
            self.Qg = N_table(alpha=ALPHA, gamma=GAMMA, mask=global_context())
            self.Qc = N_table(alpha=ALPHA, gamma=GAMMA, mask=column())
            self.Qr = N_table(alpha=ALPHA, gamma=GAMMA, mask=row())
            self.Ql = LinearNoveltyEstimator(alpha=ALPHA, gamma=GAMMA, mask=linear())
            self.Qe = CombinedAIC([self.Qs, self.Qg, self.Qr, self.Qc, self.Ql], RSS_alpha=ALPHA, weights_method=WEIGHTS_METHOD)

    def select_action(self, t, greedy=False):
        if t.action != 'initialize' and not greedy:
            reward = 1/(self.Qs.visits[t.state][t.action] + 1) if t.state_ != t.state else 0
            self.Qe.update_RSS(t.action, reward, t.state_)
        if greedy:  # use estimator with minimum RSS
            values = self.Qe.predict(t.state)
            # print(t, values)
            return np.random.choice(np.flatnonzero(values == values.max()))
        else:  # use the AIC combined estimators
            values = self.Qe.predict(t.state)
            return np.random.choice(np.flatnonzero(values == values.max()))

    def update(self, transitions):
        for t in transitions:
            self.Qe.update(t)

class Pseudocount:
    def __init__(self, dim2D=False, ALPHA=0.1, GAMMA=0.9, RSS_alpha=0.1, LIN_alpha=0.0001, WEIGHTS_METHOD='exp_size_corrected'):
        self.N = pseudoCountNovelty(features=[identity, row, column], alpha=1)
        if dim2D:
            self.Qs = N_table(alpha=ALPHA, gamma=GAMMA, mask=identity())
            self.Qg = N_table(alpha=ALPHA, gamma=GAMMA, mask=global_context())
            self.Ql = LinearNoveltyEstimator(alpha=ALPHA, gamma=GAMMA)
            self.Qe = CombinedAIC([self.Qs, self.Qg, self.Ql], RSS_alpha=ALPHA, weights_method=WEIGHTS_METHOD)
        else:
            self.Qs = N_table(alpha=ALPHA, gamma=GAMMA, mask=identity())
            self.Qg = N_table(alpha=ALPHA, gamma=GAMMA, mask=global_context())
            self.Qc = N_table(alpha=ALPHA, gamma=GAMMA, mask=column())
            self.Qr = N_table(alpha=ALPHA, gamma=GAMMA, mask=row())
            self.Qe = CombinedAIC([self.Qs, self.Qg, self.Qr, self.Qc], RSS_alpha=ALPHA, weights_method=WEIGHTS_METHOD)

    def select_action(self, t, greedy=False):
        if t.action != 'initialize' and not greedy:
            novelty = self.N.evaluate(t.state) if t.state_ != t.state else 0
            self.Qe.update_RSS(t.action, novelty, t.state_)
        if greedy:  # use estimator with minimum RSS
            values = self.Qe.predict(t.state)
            # print(t, values)
            return np.random.choice(np.flatnonzero(values == values.max()))
        else:  # use the AIC combined estimators
            values = self.Qe.predict(t.state)
            return np.random.choice(np.flatnonzero(values == values.max()))

    def update(self, transitions):
        for t in transitions:
            self.Qe.update(t)

class SimpleQ:
    def __init__(self, dim2D=False, EPSILON=0.1, ALPHA=0.1, GAMMA=0.9, RSS_alpha=0.1, LIN_alpha=0.0001, WEIGHTS_METHOD='exp_size_corrected'):
        self.epsilon = EPSILON
        if dim2D:
            self.Qs = Q_table(alpha=ALPHA, gamma=GAMMA, mask=identity())
            self.Qe = CombinedAIC([self.Qs], RSS_alpha=ALPHA, weights_method=WEIGHTS_METHOD)
        else:
            self.Qs = Q_table(alpha=ALPHA, gamma=GAMMA, mask=identity())
            self.Qe = CombinedAIC([self.Qs], RSS_alpha=ALPHA, weights_method=WEIGHTS_METHOD)

    def select_action(self, t, greedy=False):
        if t.action != 'initialize' and not greedy:
            self.Qe.update_RSS(t.action, t.reward, t.state_)

        if greedy:  # use estimator with minimum RSS
            values = self.Qe.predict(t.state)
            return np.random.choice(np.flatnonzero(values == values.max()))
        else:  # use the AIC combined estimators
            values = self.Qe.predict(t.state)
            if np.random.random() > 1 - self.epsilon:
                return np.random.randint(low=0, high=3)  # picks a random action
            else:
                return np.random.choice(np.flatnonzero(values == values.max()))  # picks the 'best' action
            # print(values, t.state)

    def update(self, transitions):
        for t in transitions:
            self.Qe.update(t)

class MEQ:
    def __init__(self, dim2D=False, EPSILON=0.1, ALPHA=0.1, GAMMA=0.9, RSS_alpha=0.1, LIN_alpha=0.0001, WEIGHTS_METHOD='exp_size_corrected'):
        self.epsilon = EPSILON
        if dim2D:
            self.Qs = Q_table(alpha=ALPHA, gamma=GAMMA, mask=identity())
            self.Qg = Q_table(alpha=ALPHA, gamma=GAMMA, mask=global_context())
            self.Qc = Q_table(alpha=ALPHA, gamma=GAMMA, mask=column())
            self.Qr = Q_table(alpha=ALPHA, gamma=GAMMA, mask=row())
            self.Ql = LinearEstimator(alpha=ALPHA, gamma=GAMMA, mask=linear())
            self.Qe = CombinedAIC([self.Qs, self.Qg, self.Qr, self.Qc, self.Ql], RSS_alpha=ALPHA, weights_method=WEIGHTS_METHOD)
        else:
            self.Qs = Q_table(alpha=ALPHA, gamma=GAMMA, mask=identity())
            self.Qg = Q_table(alpha=ALPHA, gamma=GAMMA, mask=global_context())
            self.Ql = LinearEstimator(alpha=0.0001, gamma=GAMMA, mask=linear())
            self.Qe = CombinedAIC([self.Qs, self.Qg, self.Ql], RSS_alpha=ALPHA, weights_method=WEIGHTS_METHOD)
            
    def select_action(self, t, greedy=False):
        if t.action != 'initialize' and not greedy:
            self.Qe.update_RSS(t.action, t.reward, t.state_)

        if greedy:  # use estimator with minimum RSS
            values = self.Qe.predict(t.state)
            return np.random.choice(np.flatnonzero(values == values.max()))
        else:  # use the AIC combined estimators
            if np.random.random() > 1 - self.epsilon:
                values = np.zeros(4)  # picks a random action
            else:
                values = self.Qe.predict(t.state)  # picks the 'best' action
            return np.random.choice(np.flatnonzero(values == values.max()))

    def update(self, transitions):
        for t in transitions:
            self.Qe.update(t)
            # self.Qs.update(t)

class RMAXQ:
    def __init__(self, dim2D=False, ALPHA=0.1, GAMMA=0.9, RSS_alpha=0.1, LIN_alpha=0.0001, WEIGHTS_METHOD='exp_size_corrected'):
        if dim2D:
            self.Qs = RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=1, mask=identity())
            self.Qg = RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=1, mask=global_context())
            self.Qe = CombinedAIC([self.Qs, self.Qg], RSS_alpha=ALPHA, weights_method=WEIGHTS_METHOD)
        else:
            self.Qs = RMax_table(alpha=ALPHA, gamma=GAMMA, mask=identity())
            self.Qg = RMax_table(alpha=ALPHA, gamma=GAMMA, mask=global_context())
            self.Qc = RMax_table(alpha=ALPHA, gamma=GAMMA, mask=column())
            self.Qr = RMax_table(alpha=ALPHA, gamma=GAMMA, mask=row())
            self.Qe = CombinedAIC([self.Qs, self.Qg, self.Qr, self.Qc], RSS_alpha=ALPHA, weights_method=WEIGHTS_METHOD)

    def select_action(self, t, greedy=False):
        if t.action != 'initialize':
            self.Qe.update_RSS(t.action, t.reward, t.state_)

        if greedy:  # use estimator with minimum RSS
            values = self.Qe.predict(t.state)
            return np.random.choice(np.flatnonzero(values == values.max()))
        else:  # use the AIC combined estimators
            values = self.Qe.predict(t.state)
            return np.random.choice(np.flatnonzero(values == values.max()))

    def update(self, transitions):
        for t in transitions:
            self.Qe.update(t)

class RMAXLin:
    def __init__(self, dim2D=False, ALPHA=0.1, GAMMA=0.9, RSS_alpha=0.1, LIN_alpha=0.0001, WEIGHTS_METHOD='exp_size_corrected'):
        if dim2D:
            self.Qs = RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.5, mask=identity())
            self.Qg = RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.5, mask=global_context())
            self.Ql = LinearEstimator(alpha=ALPHA, gamma=GAMMA, b=0.5)
            self.Qe = CombinedAIC([self.Qs, self.Qg], RSS_alpha=ALPHA, weights_method=WEIGHTS_METHOD)
        else:
            self.Qs = RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.5, mask=identity())
            self.Qg = RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.5, mask=global_context())
            self.Qc = RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.5, mask=column())
            self.Qr = RMax_table(alpha=ALPHA, gamma=GAMMA, MAX=0.5, mask=row())
            self.Ql = LinearEstimator(alpha=ALPHA, gamma=GAMMA, b=0.5)
            self.Qe = CombinedAIC([self.Qs, self.Qg, self.Qr, self.Qc, self.Ql], RSS_alpha=ALPHA,
                                      weights_method=WEIGHTS_METHOD)

    def select_action(self, t, greedy=False):
        if t.action != 'initialize':
            self.Qe.update_RSS(t.action, t.reward, t.state_)

        if greedy:  # use estimator with minimum RSS
            values = self.Qe.predict(t.state)
            return np.random.choice(np.flatnonzero(values == values.max()))
        else:  # use the AIC combined estimators
            values = self.Qe.predict(t.state)
            return np.random.choice(np.flatnonzero(values == values.max()))

    def update(self, transitions):
        for t in transitions:
            self.Qe.update(t)