import numpy as np

grid = np.ones((8, 8)) * 0
grid[3:6, 3:6] = -1
grid[7, 7] = 1
terminal_state = np.array([7, 7])
initial_state = np.array([1, 0])
blacked_state = np.array([[5, 5], [4, 5], [5, 4], [4, 4]])
# blacked_state = np.array([[np.nan, np.nan], [np.nan, np.nan]])

# poolGW = Gridworld(grid, terminal_state, initial_state, blacked_state)