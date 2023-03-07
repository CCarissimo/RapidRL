import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("death_experiment_shareDeath_rep10_n10000_buffer1000.csv")
print(df)

# random = []
# init0 = []
# init1 = []
#
# counter = 0
# for size in np.linspace(0, 15, 16):
#     for initial in [0, 1, 2]:
#         for iteration in range(40):
#             if initial == 0:
#                 random.append(counter)
#             elif initial == 1:
#                 init0.append(counter)
#             elif initial == 2:
#                 init1.append(counter)
#             counter += 1
#
# df["initial"] = 0
# df["initial"].loc[random] = "random"
# df["initial"].loc[init0] = "init0"
# df["initial"].loc[init1] = "init1"

size_stories_list = [0, 1, 2, 5, 7, 15]

means = df.groupby(["reset_buffer_type", "reset_q_table", "reset_visits", 'size_stories']).mean()
std = df.groupby(["reset_buffer_type", "reset_q_table", "reset_visits", "size_stories"]).std()


def plot_case(buffer, table, visits, quantity, label):
    means.loc[buffer, table, visits][quantity].plot(label=label)
    plt.fill_between(size_stories_list,
                     means.loc[buffer, table, visits][quantity] + std.loc[buffer, table, visits][quantity],
                     means.loc[buffer, table, visits][quantity] - std.loc[buffer, table, visits][quantity],
                     alpha=0.2)


reset_q_table = False
reset_visits = True

import seaborn as sns

sns.set_context("paper")
# plt.style.use('paper.mplstyle')
sns.set_palette("magma")

plt.figure(0)
plt.ylabel("Average Novelty Table Difference")
plot_case("death", reset_q_table, reset_visits, "abs_n_table_differences_mean", "death")
plot_case("random", reset_q_table, reset_visits, "abs_n_table_differences_mean", "random")
plot_case("both", reset_q_table, reset_visits, "abs_n_table_differences_mean", "both")
plt.legend()

plt.figure(1)
plt.ylabel("Average Visits Table Difference")
plot_case("death", reset_q_table, reset_visits, "abs_visits_differences_mean", "death")
plot_case("random", reset_q_table, reset_visits, "abs_visits_differences_mean", "random")
plot_case("both", reset_q_table, reset_visits, "abs_visits_differences_mean", "both")
plt.legend()

plt.figure(2)
plt.ylabel("Average Lifetime")
plot_case("death", reset_q_table, reset_visits, "lifetime_average", "death")
plot_case("random", reset_q_table, reset_visits, "lifetime_average", "random")
plot_case("both", reset_q_table, reset_visits, "lifetime_average", "both")
plt.legend()

plt.figure(3)
plt.ylabel("number_of_deaths")
plot_case("death", reset_q_table, reset_visits, "number_of_deaths", "death")
plot_case("random", reset_q_table, reset_visits, "number_of_deaths", "random")
plot_case("both", reset_q_table, reset_visits, "number_of_deaths", "both")
plt.legend()

plt.show()
