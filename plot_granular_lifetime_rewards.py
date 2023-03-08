import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("storage/merged_storage", "rb") as file:
    data = pickle.load(file)

max_steps = 10000
granularity = 101
repetitions = 10
size_stories = 1
reset_q_table = True
reset_visits = True
reset_buffer_type = "random"

D = data[(size_stories, reset_q_table, reset_visits, reset_buffer_type)]


def compute(data, size_stories, reset_q_table, reset_visits, reset_buffer_type, max_steps, granularity, repetitions):
    initial = data[(size_stories, reset_q_table, reset_visits, reset_buffer_type)]
    values = np.zeros((repetitions, granularity))
    for i in range(repetitions):
        x_vals = np.cumsum(initial[i][1][0])
        y_vals = initial[i][1][0]
        values[i] = np.interp(np.linspace(0, max_steps, granularity), x_vals, y_vals)
    values_mean = np.mean(values, axis=0)
    values_std = np.std(values, axis=0)
    values_q25 = np.quantile(values, 0.25, axis=0)
    values_q75 = np.quantile(values, 0.75, axis=0)
    return values_mean, values_std, values_q25, values_q75


lifetimes = np.zeros((repetitions, granularity))
for i in range(repetitions):
    x_vals = np.cumsum(D[i][1][0])
    y_vals = D[i][1][0]
    lifetimes[i] = np.interp(np.linspace(0, max_steps, granularity), x_vals, y_vals)
lifetimes_mean = np.mean(lifetimes, axis=0)
lifetimes_std = np.std(lifetimes, axis=0)
lifetimes_q25 = np.quantile(lifetimes, 0.25, axis=0)
lifetimes_q75 = np.quantile(lifetimes, 0.75, axis=0)

rewards = np.zeros((repetitions, granularity), dtype='float64')
for i in range(repetitions):
    x_vals = np.cumsum(D[i][1][0])
    y_vals = [np.sum(D[i][2][0][trajectory]) for trajectory in range(len(D[i][2][0]))]
    rewards[i] = np.interp(np.linspace(0, max_steps, granularity), x_vals, y_vals)
rewards_mean = np.mean(rewards, axis=0)
rewards_std = np.std(rewards, axis=0)
rewards_q25 = np.quantile(rewards, 0.25, axis=0)
rewards_q75 = np.quantile(rewards, 0.75, axis=0)

plt.figure(0)
plt.plot(np.linspace(0, max_steps, granularity), lifetimes_mean)
# plt.fill_between(np.linspace(0, max_steps, granularity), lifetimes_mean + lifetimes_std, lifetimes_mean - lifetimes_std, alpha=0.2)
plt.fill_between(np.linspace(0, max_steps, granularity), lifetimes_q75, lifetimes_q25, alpha=0.2)
plt.ylabel("Average Lifetime")
plt.title(f"sizeStories {size_stories} resetQ {reset_q_table}, resetV {reset_visits}, resetB {reset_buffer_type}")
plt.savefig(
    f"sizeStories{size_stories}_lifetime_resetQ{reset_q_table}_resetV{reset_visits}_resetB{reset_buffer_type}.pdf")

plt.figure(1)
plt.plot(np.linspace(0, max_steps, granularity), rewards_mean)
# plt.fill_between(np.linspace(0, max_steps, granularity), rewards_mean + rewards_std, rewards_mean - rewards_std, alpha=0.2)
plt.fill_between(np.linspace(0, max_steps, granularity), rewards_q75, rewards_q25, alpha=0.2)
plt.ylabel("Cumulative Reward")
plt.title(f"sizeStories {size_stories} resetQ {reset_q_table}, resetV {reset_visits}, resetB {reset_buffer_type}")
plt.savefig(
    f"sizeStories{size_stories}_reward_resetQ{reset_q_table}_resetV{reset_visits}_resetB{reset_buffer_type}.pdf")


size1_mean, size1_std, size1_q25, size1_q75 = compute(data, 1, reset_q_table, reset_visits,
                                                      reset_buffer_type, max_steps, granularity, repetitions)
size7_mean, size7_std, size7_q25, size7_q75 = compute(data, 7, reset_q_table, reset_visits,
                                                      reset_buffer_type, max_steps, granularity, repetitions)
size15_mean, size15_std, size15_q25, size15_q75 = compute(data, 15, reset_q_table, reset_visits,
                                                          reset_buffer_type, max_steps, granularity, repetitions)

plt.figure(3)
colormap = plt.get_cmap('magma')
colors = [colormap(i) for i in np.linspace(0.25, 0.75, 3)]

plt.plot(np.linspace(0, max_steps, granularity), size1_mean, label="1", c=colors[0])
plt.fill_between(np.linspace(0, max_steps, granularity), size1_q75, size1_q25, alpha=0.2, color=colors[0])
plt.plot(np.linspace(0, max_steps, granularity), size7_mean, label="7", c=colors[1])
plt.fill_between(np.linspace(0, max_steps, granularity), size7_q75, size7_q25, alpha=0.2, color=colors[1])
plt.plot(np.linspace(0, max_steps, granularity), size15_mean, label="15", c=colors[2])
plt.fill_between(np.linspace(0, max_steps, granularity), size15_q75, size15_q25, alpha=0.2, color=colors[2])

plt.legend(title="size stories", loc="upper left")
plt.ylim((0, 2000))
plt.ylabel("average lifetime (steps/agent)", **{"fontname": "Times New Roman", "fontsize": "xx-large"})
plt.xlabel("steps", **{"fontname": "Times New Roman", "fontsize": "xx-large"})
plt.title(f"Share {reset_buffer_type} stories", **{"fontname": "Times New Roman", "fontsize": "xx-large"})
plt.tight_layout()
plt.savefig(
    f"combined_lifetime_resetQ{reset_q_table}_resetV{reset_visits}_resetB{reset_buffer_type}.pdf", bbox_inches='tight')
plt.show()
