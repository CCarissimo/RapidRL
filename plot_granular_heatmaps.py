import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("storage/merged_visits", "rb") as file:
    data = pickle.load(file)

max_steps = 10000
granularity = 101
repetitions = 10
size_stories = 7
reset_q_table = True
reset_visits = True
reset_buffer_type = "random"

data = data[(size_stories, reset_q_table, reset_visits, reset_buffer_type)]

master = np.zeros((repetitions, 3, 15, 15))

for i in range(repetitions):
    tmp = np.zeros((len(data[i]), 15, 15))
    for j in range(len(data[i])):
        tmp[j] = np.array(data[i][j]).reshape(15, 15)
    master[i, 0] = tmp[0:1].sum(axis=0)
    master[i, 1] = tmp[0:int(len(data[i])/2)].sum(axis=0)
    master[i, 2] = tmp.sum(axis=0)

mean = master.mean(axis=0)

colormap = plt.get_cmap('magma')

plt.figure(0)
plt.imshow(mean[0], cmap=colormap)
plt.title("start", **{"fontname": "Times New Roman", "fontsize": "xx-large"})
plt.tight_layout()
plt.axis('off')
plt.savefig(f"heatmap_start_sizeStories{size_stories}_resetQ{reset_q_table}"
            f"_resetV{reset_visits}_resetB{reset_buffer_type}.pdf", bbox_inches='tight')

plt.figure(1)
plt.imshow(mean[1], cmap=colormap)
plt.title("middle", **{"fontname": "Times New Roman", "fontsize": "xx-large"})
plt.tight_layout()
plt.axis('off')
plt.savefig(f"heatmap_middle_sizeStories{size_stories}_resetQ{reset_q_table}"
            f"_resetV{reset_visits}_resetB{reset_buffer_type}.pdf", bbox_inches='tight')

plt.figure(2)
plt.imshow(mean[2], cmap=colormap)
plt.title("end", **{"fontname": "Times New Roman", "fontsize": "xx-large"})
plt.tight_layout()
plt.axis('off')
plt.savefig(f"heatmap_end_sizeStories{size_stories}_resetQ{reset_q_table}"
            f"_resetV{reset_visits}_resetB{reset_buffer_type}.pdf", bbox_inches='tight')

plt.show()
