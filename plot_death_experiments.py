import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("death_experiment_shareDeath_rep(40)_n(10000)_buffer(1000).csv")
print(df)

random = []
init0 = []
init1 = []

counter = 0
for size in np.linspace(0, 15, 16):
    for initial in [0, 1, 2]:
        for iteration in range(40):
            if initial == 0:
                random.append(counter)
            elif initial == 1:
                init0.append(counter)
            elif initial == 2:
                init1.append(counter)
            counter += 1

df["initial"] = 0
df["initial"].loc[random] = "random"
df["initial"].loc[init0] = "init0"
df["initial"].loc[init1] = "init1"

size_stories_list = np.linspace(0, 15, 16)

means = df.groupby(["initial", 'size_stories']).mean()
std = df.groupby(["initial", "size_stories"]).std()


def plot_case(initial, quantity):
    means.loc[initial][quantity].plot()
    plt.fill_between(size_stories_list,
                     means.loc[initial][quantity] + std.loc[initial][quantity],
                     means.loc[initial][quantity] - std.loc[initial][quantity],
                     alpha=0.2)


plt.figure(0)
plt.ylabel("Average Novelty Table Difference")
plot_case("random", "abs_n_table_differences_mean")
plot_case("init0", "abs_n_table_differences_mean")
plot_case("init1", "abs_n_table_differences_mean")

plt.figure(1)
plt.ylabel("Average Visits Table Difference")
plot_case("random", "abs_visits_differences_mean")
plot_case("init0", "abs_visits_differences_mean")
plot_case("init1", "abs_visits_differences_mean")

plt.figure(2)
plt.ylabel("Average Lifetime")
plot_case("random", "lifetime_average")
plot_case("init0", "lifetime_average")
plot_case("init1", "lifetime_average")

plt.figure(3)
plt.ylabel("number_of_deaths")
plot_case("random", "number_of_deaths")
plot_case("init0", "number_of_deaths")
plot_case("init1", "number_of_deaths")

plt.show()
