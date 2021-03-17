from abc import ABC
import numpy as np
from gym.core import Env
import copy

# class State:
#     def __init__(self, row, col):
#         self.row = row
#         self.col = col
#         self.actions_dict = {'up': np.array((-1, 0)), 'down': np.array((1, 0)), 'left': np.array((0, -1)),
#                              'right': np.array((0, 1))}
#         self.array = np.array([row, col])
#
#     def do(self, action):
#         row, col = self.array + self.actions_dict[action]
#         return State(row=row, col=col)
#
#     def __eq__(self, other):
#         return hasattr(other, 'array') and (self.array == other.array).all()
#
#     def __hash__(self):
#         return hash(str(self.array))


class Transition:
    def __init__(self, state, action, state_, reward, terminal, targets):
        self.state = state
        self.action = action
        self.state_ = state_
        self.reward = reward
        self.terminal = terminal
        self.targets = targets

    def __repr__(self):
        return f'T({self.state},{self.action})-->{self.state_})'


# Gridworld Environment class, takes grid as argument, most important method is step


class Gridworld(Env, ABC):
    def __init__(self, grid, terminal_states, initial_state, blacked_states, max_steps):
        super().__init__()
        self.actions = ['up', 'down', 'left', 'right']
        self.actions_dict = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1),
                             'right': (0, 1)}
        self.actions_map = {action: i for i, action in enumerate(self.actions)}
        self.grid = grid
        self.terminal_states = terminal_states
        self.initial_state = tuple(initial_state)
        self.blacked_states = blacked_states
        self.step_counter = 0
        self.max_steps = max_steps
        self.terminal = False
        self.transition = Transition(state=self.initial_state,
                                     action='initialize',
                                     state_=self.initial_state,
                                     reward=self.grid[self.initial_state],
                                     terminal=False,
                                     targets=None)
        self.grid_width = len(self.grid[0, :]) - 1
        self.grid_height = len(self.grid[:, 0]) - 1

    def act_from_state(self, state, action):
        return tuple(map(lambda x, y: x + y, state, self.actions_dict[action]))

    def step(self, action):
        state_ = self.act_from_state(self.transition.state_, action)

        if state_[0] < 0 or state_[0] > self.grid_height:
            state_ = self.transition.state_
        elif state_[1] < 0 or state_[1] > self.grid_width:
            state_ = self.transition.state_
        else:
            for s in self.blacked_states:
                if (state_ == s).all():
                    state_ = self.transition.state_
                    continue

        reward = self.grid[state_]

        for s in self.terminal_states:
            if (state_ == s).all():
                self.terminal = True

        if self.step_counter == self.max_steps:
            self.terminal = True

        self.transition = Transition(state=self.transition.state_, action=action, state_=state_, reward=reward,
                                     terminal=self.terminal, targets=None)

        self.step_counter += 1

        return copy.deepcopy(self.transition)

    def set(self, state: tuple):
        self.transition = Transition(state=state, action='initialize', state_=state,
                                     reward=self.grid[state],
                                     terminal=False, targets=None)
        self.terminal = False
        self.step_counter = 0
        return copy.deepcopy(self.transition)

    def reset(self):
        self.transition = Transition(state=self.initial_state, action='initialize', state_=self.initial_state,
                                     reward=self.grid[self.initial_state],
                                     terminal=False, targets=None)
        self.terminal = False
        self.step_counter = 0
        return copy.deepcopy(self.transition)
