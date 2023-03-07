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

data = data[(size_stories, reset_q_table, reset_visits, reset_buffer_type)]

lifetimes = np.zeros((repetitions, granularity))
for i in range(repetitions):
    x_vals = np.cumsum(data[i][1][0])
    y_vals = data[i][1][0]
    lifetimes[i] = np.interp(np.linspace(0, max_steps, granularity), x_vals, y_vals)
lifetimes_mean = np.mean(lifetimes, axis=0)
lifetimes_std = np.std(lifetimes, axis=0)

rewards = np.zeros((repetitions, granularity), dtype='float64')
for i in range(repetitions):
    x_vals = np.cumsum(data[i][1][0])
    y_vals = [np.sum(data[i][2][0][trajectory]) for trajectory in range(len(data[i][2][0]))]
    rewards[i] = np.interp(np.linspace(0, max_steps, granularity), x_vals, y_vals)
rewards_mean = np.mean(rewards, axis=0)
rewards_std = np.std(rewards, axis=0)

plt.figure(0)
plt.plot(np.linspace(0, max_steps, granularity), lifetimes_mean)
plt.fill_between(np.linspace(0, max_steps, granularity), lifetimes_mean + lifetimes_std, lifetimes_mean - lifetimes_std, alpha=0.2)
plt.ylabel("Average Lifetime")

plt.figure(1)
plt.plot(np.linspace(0, max_steps, granularity), rewards_mean)
plt.fill_between(np.linspace(0, max_steps, granularity), rewards_mean + rewards_std, rewards_mean - rewards_std, alpha=0.2)
plt.ylabel("Cumulative Reward")

plt.show()
