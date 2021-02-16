from abc import ABC
import numpy as np
from gym.core import Env
import matplotlib.pyplot as plt


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


# Gridworld Environment class, takes grid as argument, most important method is step
class gridworld(Env, ABC):
    def __init__(self, grid, terminal_states, initial_state, blacked_states, max_steps):
        super().__init__()
        self.actions = ['up', 'down', 'left', 'right']
        self.actions_dict = {'up': np.array((-1, 0)), 'down': np.array((1, 0)), 'left': np.array((0, -1)),
                             'right': np.array((0, 1))}
        self.actions_map = {action: i for i, action in enumerate(self.actions)}
        self.grid = grid
        self.terminal_states = terminal_states
        self.initial_state = initial_state
        self.blacked_states = blacked_states
        self.step_counter = 0
        self.max_steps = max_steps
        self.terminal = False
        self.state = self.initial_state
        self.grid_width = len(self.grid[0, :]) - 1
        self.grid_height = len(self.grid[:, 0]) - 1

    def step(self, action):

        state_ = self.state + self.actions_dict[action]

        if state_[0] < 0 or state_[0] > self.grid_height:
            state_ = self.state
        elif state_[1] < 0 or state_[1] > self.grid_width:
            state_ = self.state
        else:
            for s in self.blacked_states:
                if (state_ == s).all():
                    state_ = self.state
                    continue

        reward = self.grid[state_[0], state_[1]]

        for s in self.terminal_states:
            if (state_ == s).all():
                self.terminal = True

        if self.step_counter == self.max_steps:
            self.terminal = True

        t_sa = Transition(state=self.state, action=action, state_=state_, reward=reward, terminal=self.terminal,
                          targets=None)
        self.state = state_
        self.step_counter += 1

        return t_sa

    def reset(self):
        self.state = self.initial_state
        self.terminal = False
        self.step_counter = 0
        return self.state