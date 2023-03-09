import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("storage/merged_visits", "rb") as file:
    data = pickle.load(file)

max_steps = 10000
granularity = 101
repetitions = 10
size_stories = 1
reset_q_table = True
reset_visits = True
reset_buffer_type = "random"

np.random.seed(seed=0)
x_random_death_states = np.random.randint(1, 15, size=10)
np.random.seed(seed=1)
y_random_death_states = np.random.randint(1, 15, size=10)
terminal_state = set(zip(y_random_death_states, x_random_death_states))

data = data[(size_stories, reset_q_table, reset_visits, reset_buffer_type)]

master = np.zeros((repetitions, 3, 15, 15))
visits = np.zeros((repetitions, 3, 15, 15))

for i in range(repetitions):
    tmp = np.zeros((len(data[i]), 15, 15))
    for j in range(len(data[i])):
        tmp[j] = np.array(data[i][j]).reshape(15, 15).T

    master[i, 0] = tmp[0].T
    master[i, 1] = tmp[int(len(data[i])/2)].T
    master[i, 2] = tmp[-1].T

    visits[i, 0] = tmp[0]
    visits[i, 1] = tmp[int(len(data[i])/2)]
    visits[i, 2] = tmp[-1]

mean = master.mean(axis=0)
visits = visits.mean(axis=0).astype(float)
print(visits)

colormap = plt.get_cmap('magma')

fig, ax = plt.subplots(ncols=3)

ax[0].imshow(mean[0], cmap=colormap)
ax[1].imshow(mean[1], cmap=colormap)
ax[2].imshow(mean[2], cmap=colormap)

ax[0].yaxis.set_ticklabels([])
ax[0].xaxis.set_ticklabels([])
ax[0].yaxis.set_ticks([])
ax[0].xaxis.set_ticks([])
ax[0].set_ylabel(reset_buffer_type, **{"fontname": "Times New Roman", "fontsize": "x-large"})
ax[1].axis("off")
ax[2].axis("off")

# ax[0].set_title("first", **{"fontname": "Times New Roman", "fontsize": "x-large"})
# ax[1].set_title("middle", **{"fontname": "Times New Roman", "fontsize": "x-large"})
# ax[2].set_title("last", **{"fontname": "Times New Roman", "fontsize": "x-large"})

for (i, j) in terminal_state:
    if visits[0, j, i] > 0:
        ax[0].text(j, i, 'X', ha="center", va="center", color="w", fontsize=8, **{"fontname": "Times New Roman"})
for (i, j) in terminal_state:
    if visits[1, j, i] > 0:
        ax[1].text(j, i, 'X', ha="center", va="center", color="w", fontsize=8, **{"fontname": "Times New Roman"})
for (i, j) in terminal_state:
    if visits[2, j, i] > 0:
        ax[2].text(j, i, 'X', ha="center", va="center", color="w", fontsize=8, **{"fontname": "Times New Roman"})

plt.tight_layout()
plt.savefig(f"heatmaps_sizeStories{size_stories}_resetQ{reset_q_table}"
            f"_resetV{reset_visits}_resetB{reset_buffer_type}.pdf", bbox_inches='tight')
plt.show()
