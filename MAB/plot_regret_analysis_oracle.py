import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.signal import savgol_filter

savepath = 'results\\'
deterministic = 'neither'
n_colors = 1
# name = 'det_' + deterministic + '_cum_regret_oracle' + '_%i_colors' % n_colors
name = 'avg_delta_naive_vs_oracle'

deltas = np.load('%s%s.npy' % (savepath, name))
smooth_deltas = savgol_filter(deltas[1,:].flatten(), 201, 2)

print(deltas[0,:].flatten(), deltas[1,:].flatten())
max_iter = 1000
epsilons = [0.05]

font = {'family': 'arial',
        'weight': 'normal',
        'size': 20}
matplotlib.rc('font', **font)

fig, ax1 = plt.subplots()
colormap = plt.get_cmap('RdYlBu')
color = [colormap(k) for k in np.linspace(0, 1, 100)]

ax1.set_xlabel('t')
ax1.set_ylabel(r'Simple Regret $\Delta$')
ax1.plot(range(max_iter), deltas[0,:].flatten(), label='Oracle', c=color[0], alpha=1)
ax1.plot(range(max_iter), deltas[1,:].flatten(), c=color[-1], alpha=0.3)
ax1.plot(range(max_iter), smooth_deltas, label=r'$\epsilon$-NCB', c=color[-10])
ax1.legend(bbox_to_anchor=(1,1), fontsize=16, frameon=False)
# plt.title('Deterministic %s Cumulative Regret vs Oracle' % deterministic)
plt.ylim((-0.1, np.max(deltas)+0.1))
plt.tight_layout()
plt.savefig('%s%s.png' % (savepath, name))
plt.show()