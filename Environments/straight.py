import numpy as np

grid = np.ones((3, 9)) * -1
grid[1, :8] = 0
grid[1, 8] = 1
grid[2, :8] = 0.1
terminal_state = []
for i in [0]:
    for j in range(8):
        terminal_state.append([i, j])
terminal_state.append([1, 8])
# terminal_state.append([0, 8])
# terminal_state.append([2, 8])
terminal_state = np.array(terminal_state)
initial_state = np.array([1, 0])
blacked_state = np.array([[0, 8], [2, 8]])