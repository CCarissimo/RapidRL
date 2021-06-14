from abc import ABC


class Mask(ABC):
    def __init__(self):
        pass

    def apply(self, state):
        pass


class identity(Mask):
    def apply(self, state):
        context = state
        return context

class linear(Mask):
    def apply(self, state):
        context = state
        return context

class arrival_state(Mask):
    def apply(self, transition):
        context = transition.state_
        return context


class global_context(Mask):
    def apply(self, state):
        context = 0
        return context


class column(Mask):
    def apply(self, state):
        return state[1]


class row(Mask):
    def apply(self, state):
        return state[0]
