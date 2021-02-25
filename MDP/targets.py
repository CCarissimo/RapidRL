from abc import ABC


class Target(ABC):
    def __init__(self):
        pass

    def compute(self, trajectory):
        pass


class TD0(Target):
    def __init__(self, gamma):
        self.gamma = gamma

    def compute(self, trajectory):
        nV =