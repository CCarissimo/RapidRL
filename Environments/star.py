import numpy as np
from .environments import Gridworld

grid = np.ones((15, 15)) * 0.1
grid[7, 0] = -1
grid[14, 7] = -1
grid[7, 14] = 1
grid[0, 7] = 1
terminal_state = np.array([[7,0], [14,7], [7,14], [0,7]])
initial_state = np.array([7,7])
blacked_state = np.array([np.nan, np.nan])

# starGW = Gridworld(grid, terminal_state, initial_state, blacked_state)

# if __name__ == "__main__":
	# _, _, _, _ = plot_gridworld(grid, terminal_state, initial_state, blacked_state)