import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.signal import savgol_filter

savepath = 'results\\'
name1 = 'avg_delta_naive_n&epsilon'
name2 = 'avg_arms_naive_n&epsilon'

max_iter = 1000
epsilons = 1 - np.array([0.5, 0.75, 0.95, 0.98, 0.99, 0.999])

delta_averages = np.load('%s%s.npy' % (savepath, name1))
arm_counts = np.load('%s%s.npy' % (savepath, name2))

font = {'family': 'arial',
        'weight': 'normal',
        'size': 20}
matplotlib.rc('font', **font)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


fig, ax = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6,8))
colormap = plt.get_cmap('RdYlBu')
color = [colormap(k) for k in np.linspace(0, 1, len(arm_counts[:, 0]))]

ax[0].set_ylabel(r'Simple Regret $\Delta$')
for k, epsilon in enumerate(epsilons):
    smooth_delta = savgol_filter(delta_averages[k], 201, 3)
    ax[0].scatter(range(max_iter), delta_averages[k], color=color[k], s=1, alpha=0.3)
    ax[0].plot(range(len(smooth_delta)), smooth_delta,
               label=r'$\epsilon = $%.3f' % (epsilon), color=color[k])

ax[0].legend(bbox_to_anchor=(1.01, 1.2), fontsize=12, ncol=3)

ax[1].set_ylabel('arms')
for k, epsilon in enumerate(epsilons):
    ax[1].step(range(max_iter), arm_counts[k],
               ls='--', color=color[k])
ax[1].set_xlabel('t')

plt.tight_layout()
plt.savefig('%s%s.png' % (savepath, name1))
plt.show()
