class Estimator:
    def __init__(self, approximator, mask):
        self.approximator = approximator
        self.mask = mask
        self.visits = {}

    def evaluate(self, state, action):
        c = self.mask.apply(state)
        return self.approximator.value(c, action)

    def update(self, buffer_sample):
        for i, transition in enumerate(buffer_sample):
            c = self.mask.apply(transition.state)
            self.visits[(c, transition.action)] += 1
            N_ca = self.visits[(c, transition.action)]
            transition.context = c
            transition.N_ca = N_ca
            buffer_sample[i] = transition

        self.approximator.update(buffer_sample)


class Mask:
    def __init__(self):
        pass

    def apply(self, state):
        pass


class identity(Mask):
    def apply(self, state):
        return state


class Approximator:
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
        self.table = {}

    def update_table(self, transition):
        if transition.state not in self.table.keys():
            self.table[transition.state] = {a: 0 for a in self.actions}
        if transition.state_ not in self.table.keys():
            self.table[transition.state_] = {a: 0 for a in self.actions}

    def update_value(self, transition):
        pass

    def update(self, buffer_sample):
        for transition in buffer_sample:
            self.update_table(transition)
            self.update_value(transition)

    def evaluate(self, c, a):
        return self.table[c][a]


# class counter_table(table):
#     def update_value(self, s, a, s_, r, targets, N_ca):


class bellman_table(table):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def update_value(self, t):
        self.table[t.state][t.action] = self.table[t.state][t.action] + self.alpha * (t.reward + self.gamma * max(
            v for v in self.table[t.state_].values()) - self.table[t.state][t.action])


class average(table):
    def update_value(self, t):
        self.table[t.context][t.action] = self.table[t.context][t.action] * (t.N_ca - 1)/t.N_ca + t.reward * 1/t.N_ca
