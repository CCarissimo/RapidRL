class Estimator:
    def __init__(self, approximator, mask):
        self.approximator = approximator
        self.mask = mask
        self.visits = {}

    def Evaluate(self, state, action):
        c = self.mask(state)
        return self.approximator.value(c, action)

    def Update(self, buffer_sample):
        for i, transition in enumerate(buffer_sample):
            s, a, s_, r, _, targets = transition
            c = self.mask(s)
            self.visits[(c, a)] += 1
            N_ca = self.visits[(c, a)]
            buffer_sample[i] = (c, a, s_, r, targets, N_ca)

        self.approximator.update(buffer_sample)


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

    def update_table(self, s, s_):
        if s not in self.table.keys():
            self.table[s] = {a: 0 for a in self.actions}
        if s_ not in self.table.keys():
            self.table[s_] = {a: 0 for a in self.actions}

    def update_value(self, s, a, s_, r, targets, N_ca):
        self.table[s][a] = r

    def update(self, buffer_sample):
        for transition in buffer_sample:
            s, a, s_, r, targets, N_ca = transition

            self.update_table(s, s_)

            self.update_value(s, a, s_, r, targets, N_ca)

    def evaluate(self, c, a):
        return self.table[c][a]


class bellman_table(table):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def update_value(self, s, a, s_, r, targets, N_ca):
        self.table[s][a] = self.table[s][a] + self.alpha * (r + self.gamma * max(
            v for v in self.table[s_].values()) - self.table[s][a])


class average(table):
    def update_value(self, c, a, s_, r, targets, N_ca):
        self.table[c][a] = self.table[c][a] * (N_ca - 1)/N_ca + r * 1/N_ca


# class N(Estimator):
#     def update(self, transition, alpha):
#         s, a, s_, r, _, targets = transition
#         s = str(s)
#         s_ = str(s_)
#         if s not in self.value.keys():
#             self.value[s] = {action: 0 for action in self.actions}
#         if s_ not in self.value.keys():
#             self.value[s_] = {action: 0 for action in self.actions}
#
#         self.value[s][a] = self.value[s][a] + alpha * (r + self.gamma * max(
#             v for v in self.value[s_].values()) - self.value[s][a])
