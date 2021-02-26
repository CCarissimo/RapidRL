import numpy as np

class Agent:
    def __init__(self, estimators, buffer, targets, counter, batch_size=1):
        self.actions = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.actions)
        self.actions_map = {action: i for i, action in enumerate(self.actions)}
        self.trajectory = []
        self.batch_size = batch_size
        self.estimators = estimators
        self.buffer = buffer
        self.targets = targets
        self.counter = counter

    def select_action(self, transition):
        self.process_state_estimators(transition)
        action = self.combine_state_estimators(transition)
        return action

    def observe(self, transition):
        self.trajectory.append(transition)
        self.update_counter(transition)
        if transition.terminal:
            T = self.compute_trajectory_targets(self.trajectory)
            self.store_trajectory(self.trajectory, T)

    def process_state_estimators(self, transition):
        if self.buffer.size > self.batch_size:
            S = self.buffer.sample(self.batch_size)
            for E in self.estimators:
                E.update(S)

    def combine_state_estimators(self, transition):
        action = self.actions[np.random.randint(self.n_actions)]
        return action

    def update_counter(self, transition):
        self.counter.update_table(transition)
        self.counter.update_value(transition)

    def compute_trajectory_targets(self, trajectory):
        for target in self.targets:
            target.compute(trajectory)
        return 0

    def store_trajectory(self, trajectory, targets):
        for transition in trajectory:
            transition.targets = targets
            self.buffer.append(transition)

    def reset_trajectory(self):
        self.trajectory = []


class Greedy(Agent):
    def combine_state_estimators(self, transition):
        maximizing_actions = []
        for E in self.estimators:
            action_values = E.evaluate(transition)
            maximizing_actions.append(max(action_values, key=action_values.get))
        return maximizing_actions[np.random.randint(len(maximizing_actions))]
