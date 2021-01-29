import numpy as np
import matplotlib.pyplot as plt
import matplotlib

savepath = 'results\\'
name = 'simple_regret_over_epsilon'

deltas = np.load('%s%s.npy' % (savepath, name))
search_params = np.concatenate((np.linspace(0, 0.1, 100), np.linspace(0.11, 1, 100)))

font = {'family': 'arial',
        'weight': 'normal',
        'size': 20}
matplotlib.rc('font', **font)

colormap = plt.get_cmap('RdYlBu')
color = [colormap(k) for k in np.linspace(0, 1, 2)]

plt.plot(search_params, deltas, label='Suboptimality Gap', c=color[1])
plt.scatter(search_params[np.argmin(deltas)], np.min(deltas), marker="*", s=200, c=color[0], label=r'optimal $\epsilon$')
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'Simple Regret $\Delta$')
plt.xscale('log')
# plt.legend()
plt.tight_layout()
plt.savefig('%s%s.png' % (savepath, name))
plt.show()
