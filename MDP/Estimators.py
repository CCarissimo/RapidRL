from abc import ABC


class Estimator:
    def __init__(self, approximator, mask, actions=None):
        if actions is None:
            self.actions = ['up', 'down', 'left', 'right']
        self.approximator = approximator
        self.mask = mask
        self.visits = dict()

    def evaluate(self, transition):
        c = self.mask.apply(transition)
        return self.approximator.evaluate(c, transition)

    def update(self, buffer_sample):
        for transition in buffer_sample:
            transition.N_ca = self.visits[transition.context][transition.action]
        self.approximator.update(buffer_sample)

    def count_visits(self, transition):
        transition = self.mask.apply(transition)
        if transition.context not in self.visits.keys():
            self.visits[transition.context] = {a: 0 for a in self.actions}
        self.visits[transition.context][transition.action] += 1


class Mask(ABC):
    def __init__(self):
        pass

    def apply(self, transition):
        pass


class identity(Mask):
    def apply(self, transition):
        transition.context = transition.state
        return transition


class arrival_state(Mask):
    def apply(self, transition):
        transition.context = transition.state_
        return transition


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
        self.table = dict()

    def update(self, buffer_sample):
        for transition in buffer_sample:
            self.update_table(transition)
            self.update_value(transition)

    def update_table(self, transition):
        if transition.state not in self.table.keys():
            self.table[transition.state] = {a: 0 for a in self.actions}
        if transition.state_ not in self.table.keys():
            self.table[transition.state_] = {a: 0 for a in self.actions}

    def update_value(self, transition):
        self.table[transition.state][transition.action] += 1

    def evaluate(self, c, a):
        return self.table[c][a]


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


class bellman_N_table(bellman_Q_table):
    def update_value(self, t):
        self.table[t.state][t.action] = self.table[t.state][t.action] + self.alpha * (1 / t.N_ca + self.gamma * max(
            v for v in self.table[t.state_].values()) - self.table[t.state][t.action])


class average(table):
    def update_value(self, t):
        self.table[t.context][t.action] = self.table[t.context][t.action] * (
                    t.N_ca - 1) / t.N_ca + t.reward * 1 / t.N_ca
