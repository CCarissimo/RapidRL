import numpy as np
from gym.core import Env
import matplotlib.pyplot as plt

# Gridworld Environment class, takes grid as argument, most important method is step
class gridworld(Env):
    def __init__(self, grid, terminal_states, initial_state, agent, max_steps):
        super().__init__()
        self.actions = ['up', 'down', 'left', 'right']
        self.actions_dict = {'up':np.array((-1,0)), 'down':np.array((1,0)), 'left':np.array((0,-1)), 'right':np.array((0,1))}
        self.actions_map = {action:i for i, action in enumerate(self.actions)}
        self.grid = grid
        self.terminal_states = terminal_states
        self.initial_state = initial_state
        self.step_counter = 0
        self.max_steps = max_steps
        
    def step(self, action):
        
        new_state = self.state + self.actions_dict[action]
        
        if new_state[0] < 0 or new_state[0] > 2:
            new_state = self.state
        elif new_state[1] < 0 or new_state[1] > 3:
            new_state = self.state
        elif new_state[0] == 0 and new_state[1] == 2:
            new_state = self.state
        
        self.state = new_state
        
        reward = self.grid[self.state[0], self.state[1]]
        
        observation_ = self.state
        
        for square in self.terminal_states:
            if (self.state == square).all():
                self.terminal = True
                
        if self.step_counter == self.max_steps:
            self.terminal = True
        
        self.step_counter += 1
        
        return self.state, reward, self.terminal
    
    def reset(self):
        self.state = self.initial_state
        self.terminal = False
        self.step_counter = 0
        return self.state

# Agent class, currently only selects random actions
class agent:
    def __init__(self,):
        self.actions = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.actions)
        
    def next_action(self, state):
        next_action = self.actions[np.random.randint(self.n_actions)]
        return next_action

# For (s,a), find the cumulative discounted sum of rewards over a set of trajectories
def Qpi_sa(state, action, trajectories, gamma, q_table):
    returns = []
    for t in trajectories:
        store = False
        step_counter = 0
        for s, a in t:
            disc_ret = 0
            if (state == s).all() and action == a:
                store = True
                
            if store:
                disc_ret += gamma**(step_counter) * q_table[str(s)][a]
                step_counter += 1
        
        if store:
            returns.append(disc_ret)
        
    return np.mean(returns)

# Create a dictionary indexed by state action strings with discounted 
def return_table(trajectories, gamma, q_table):
    returns_table = {}
    for t in trajectories:
        for s, a in t:
            key = str(s)+', '+a
            if key in returns_table.keys():
                continue
            else:
                returns_table[key] = Qpi_sa(s,a,trajectories,gamma,q_table)
    
    return returns_table

# Calculate state visit distribution
def state_dist(visits):
    total_visits = sum(visits[state] for state in visits.keys())
    rho = {}
    for s in visits.keys():
        rho[s] = visits[s]/total_visits
    return rho

# Using the table of returns for (s,a) pairs, abstract over states to estimate generalized (a) values
def Qpi_a(action, returns_table, rho):
    abstraction = 0
    for state, prob in rho.items():
        key = state + ', ' + action
        if key in returns_table.keys():
            abstraction += prob * returns_table[key]
    return abstraction

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones