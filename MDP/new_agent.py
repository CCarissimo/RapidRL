import numpy as np


class Agent:
    def __init__(self, buffer, targets, estimators, batch_size=1):
        self.actions = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.actions)
        self.actions_map = {action: i for i, action in enumerate(self.actions)}
        self.trajectory = []
        self.buffer = buffer
        self.targets = targets
        self.estimators = estimators
        self.batch_size = batch_size

    def select_action(self, state):
        self.process_state_estimators(state)
        action = self.combine_state_estimators(state)
        return action

    def observe(self, transition):

        self.trajectory.append(transition)

        if transition.terminal:
            self.compute_trajectory_targets(self.trajectory)
            self.store_trajectory(self.trajectory, self.targets)

    def process_state_estimators(self, state):
        if self.buffer.size > self.batch_size:
            S = self.buffer.sample(self.batch_size)
            for E in self.estimators:
                E.update(S)

    def combine_state_estimators(self, state):
        action = self.actions[np.random.randint(self.n_actions)]
        return action

    def compute_trajectory_targets(self, trajectory):
        for target in self.targets:
            target.compute(trajectory)

    def store_trajectory(self, trajectory, targets):
        for transition in trajectory:
            transition.targets = targets
            self.buffer.append(transition)

    def reset_trajectory(self):
        self.trajectory = []
