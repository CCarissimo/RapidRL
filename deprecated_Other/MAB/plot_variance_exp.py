import numpy as np
import matplotlib.pyplot as plt
import matplotlib

savepath = 'results\\'
name = 'simple_regret_over_epsilon_x_variance'

data = np.load('%s%s.npy' % (savepath, name))

epsilons = np.linspace(0, 1, 51)
best_epsilons = np.argmin(data, axis=0)
variances = np.linspace(0, 1, 51)

print(data)
print(best_epsilons)

font = {'family': 'arial',
        'weight': 'normal',
        'size': 20}
matplotlib.rc('font', **font)

colormap = plt.get_cmap('RdYlBu_r')
color = [colormap(k) for k in np.linspace(0, 1, 2)]

fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [1, 1]}, figsize=(6,12))
cax = fig.add_axes([0.27, 0.9, 0.5, 0.05])

im = ax1.contourf(data, extent=[0,1,0,1], cmap = colormap)
ax1.set_ylabel(r'$\epsilon$')
#plt.xlabel(r'$(\sigma_c^2, 1-\sigma_a^2)$')
cb = fig.colorbar(im, ax=ax1, cax=cax, orientation='horizontal')
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_label_position('top')
cb.ax.tick_params(labelsize=10)
#plt.tight_layout()
#plt.savefig('%s%s_contourf.png' % (savepath, name))
#plt.show()


ax2.plot(variances, epsilons[best_epsilons], color=color[1])
ax2.set_ylabel(r'$\epsilon^*$')
ax2.set_xlabel(r'$(\sigma_c^2, 1-\sigma_a^2)$')
ax2.set_ylim((0,1))
ax2.set_xlim((0,1))

#plt.tight_layout()
# plt.savefig('%s%s.png' % (savepath, name))
plt.show()