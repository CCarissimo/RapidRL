class State:
    def __init__(self):
        pass


class Transition:
    def __init__(self, state, action, state_, reward, terminal, targets):
        self.state = state
        self.action = action
        self.state_ = state_
        self.reward = reward
        self.terminal = terminal
        self.targets = targets

    def unpack_transition(self):
        transition = (self.state, self.action, self.state_, self.reward, self.terminal)
        return transition

