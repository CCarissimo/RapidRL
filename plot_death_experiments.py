import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("death_experiment_n[10000]_alpha[0.1]_gamma[0.2]_batch[10].csv")
print(df)

buffer_size_list = np.linspace(100, 10000, 31)

means = df.groupby(['buffer_size']).mean()
std = df.groupby(["buffer_size"]).std()

plt.figure(0)
plt.ylabel("Average Novelty Table Difference")
means["n_table_differences_mean"].plot()
plt.fill_between(buffer_size_list,
                 means["n_table_differences_mean"] + std["n_table_differences_mean"],
                 means["n_table_differences_mean"] - std["n_table_differences_mean"],
                 alpha=0.2)

plt.figure(1)
plt.ylabel("Average Visits Table Difference")
means["visits_differences_mean"].plot()
plt.fill_between(buffer_size_list,
                 means["visits_differences_mean"] + std["visits_differences_mean"],
                 means["visits_differences_mean"] - std["visits_differences_mean"],
                 alpha=0.2)

plt.figure(2)
plt.ylabel("Average Lifetime")
means["lifetime_average"].plot()
plt.fill_between(buffer_size_list,
                 means["lifetime_average"] + std["lifetime_average"],
                 means["lifetime_average"] - std["lifetime_average"],
                 alpha=0.2)

plt.figure(3)
plt.ylabel("number_of_deaths")
means["number_of_deaths"].plot()
plt.fill_between(buffer_size_list,
                 means["number_of_deaths"] + std["number_of_deaths"],
                 means["number_of_deaths"] - std["number_of_deaths"],
                 alpha=0.2)

plt.show()
