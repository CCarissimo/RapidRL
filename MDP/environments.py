from abc import ABC
import numpy as np
from gym.core import Env


class State:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.actions_dict = {'up': np.array((-1, 0)), 'down': np.array((1, 0)), 'left': np.array((0, -1)),
                             'right': np.array((0, 1))}
        self.array = np.array([row, col])

    def do(self, action):
        row, col = self.array + self.actions_dict[action]
        return State(row=row, col=col)

    def __eq__(self, other):
        return hasattr(other, 'array') and (self.array == other.array).all()

    def __hash__(self):
        return hash(str(self.array))


class Transition:
    def __init__(self, state, action, state_, reward, terminal, targets):
        self.state = state
        self.action = action
        self.state_ = state_
        self.reward = reward
        self.terminal = terminal
        self.targets = targets


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
        self.initial_state = State(row=initial_state[0], col=initial_state[1])
        self.blacked_states = blacked_states
        self.step_counter = 0
        self.max_steps = max_steps
        self.terminal = False
        self.state = State(row=initial_state[0], col=initial_state[1])
        self.grid_width = len(self.grid[0, :]) - 1
        self.grid_height = len(self.grid[:, 0]) - 1

    def step(self, action):

        state_ = self.state.do(action)

        if state_.row < 0 or state_.row > self.grid_height:
            state_ = self.state
        elif state_.col < 0 or state_.col > self.grid_width:
            state_ = self.state
        else:
            for s in self.blacked_states:
                if (state_.array == s).all():
                    state_ = self.state
                    continue

        reward = self.grid[state_.row, state_.col]

        for s in self.terminal_states:
            if (state_.array == s).all():
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
