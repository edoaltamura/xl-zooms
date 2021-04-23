from merge_catalogues import catalogue, select_runs, models
from matplotlib import pyplot as plt
from unyt import kb

import sys

sys.path.append("..")

from literature import Sun2009


try:
    plt.style.use("../register/mnras.mplstyle")
except:
    pass


fig = plt.figure(figsize=(7, 5))
gs = fig.add_gridspec(2, 3, hspace=0.2, wspace=0.)
axes = gs.subplots(sharex=True, sharey=True)


for ax in axes.flat:
    ax.loglog()

# 'Run_name', 'm_gas', 'f_gas', 'T500', 'T2500', 'T500_nocore',
#        'T2500_nocore', 'r2500', 'r1000', 'r500', 'r200', 'm2500', 'm1000',
#        'm500', 'm200', 'm_star500c', 'm_star30kpc', 'm_star', 'f_star',
#        'k30kpc', 'k2500', 'k1500', 'k1000', 'k500', 'k200', 'Ekin', 'Etherm',
#        'r_1500', 'm_1500', 'LX500', 'LX2500', 'LX500_nocore', 'LX2500_nocore'

axes[0, 0].set_ylabel('Entropy [keV cm$^2$]')
axes[0, 0].set_xlabel('$k_BT_{500}^{>0.15 r_{500}}$ [keV]')
axes[0, 1].set_xlabel('$k_BT_{500}^{>0.15 r_{500}}$ [keV]')
axes[0, 2].set_xlabel('$k_BT_{500}^{>0.15 r_{500}}$ [keV]')

axes[1, 0].set_ylabel('Entropy [keV cm$^2$]')
axes[1, 0].set_xlabel('$k_BT_{2500}^{>0.15 r_{500}}$ [keV]')
axes[1, 1].set_xlabel('$k_BT_{2500}^{>0.15 r_{500}}$ [keV]')
axes[1, 2].set_xlabel('$k_BT_{2500}^{>0.15 r_{500}}$ [keV]')

for model in models:
    df = catalogue[catalogue['Run_name'].str.contains(model, regex=False)]

    axes[0, 0].scatter([(t * kb).to('keV') for t in df['T500_nocore']], df['k500'], s=1)
    axes[0, 1].scatter([(t * kb).to('keV') for t in df['T500_nocore']], df['k1000'], s=1)
    axes[0, 2].scatter([(t * kb).to('keV') for t in df['T500_nocore']], df['k1500'], s=1)
    axes[1, 0].scatter([(t * kb).to('keV') for t in df['T2500_nocore']], df['k2500'], s=1)
    axes[1, 1].scatter([(t * kb).to('keV') for t in df['T2500_nocore']], df['k0p15r500'], s=1)
    axes[1, 2].scatter([(t * kb).to('keV') for t in df['T2500_nocore']], df['k30kpc'], s=1)

    axes[0, 0].text(0.03, 0.97, r'$K_{500}$', horizontalalignment='left', verticalalignment='top', transform=axes[0, 0].transAxes)
    axes[0, 1].text(0.03, 0.97, r'$K_{1000}$', horizontalalignment='left', verticalalignment='top', transform=axes[0, 1].transAxes)
    axes[0, 2].text(0.03, 0.97, r'$K_{1500}$', horizontalalignment='left', verticalalignment='top', transform=axes[0, 2].transAxes)
    axes[1, 0].text(0.03, 0.97, r'$K_{2500}$', horizontalalignment='left', verticalalignment='top', transform=axes[1, 0].transAxes)
    axes[1, 1].text(0.03, 0.97, r'$K_{0.15r500}$', horizontalalignment='left', verticalalignment='top', transform=axes[1, 1].transAxes)
    axes[1, 2].text(0.03, 0.97, r'$K_{30 \rm kpc}$', horizontalalignment='left', verticalalignment='top', transform=axes[1, 2].transAxes)

sun2009 = Sun2009()
axes[0, 0].scatter(sun2009.T_500, sun2009.K_500, color='k', marker='*', edgecolors='none')
axes[0, 1].scatter(sun2009.T_500, sun2009.K_1000, color='k', marker='*', edgecolors='none')
axes[0, 2].scatter(sun2009.T_500, sun2009.K_1500, color='k', marker='*', edgecolors='none')
axes[1, 0].scatter(sun2009.T_2500, sun2009.K_2500, color='k', marker='*', edgecolors='none')
axes[1, 1].scatter(sun2009.T_2500, sun2009.K_0p15r500, color='k', marker='*', edgecolors='none')
axes[1, 2].scatter(sun2009.T_2500, sun2009.K_30kpc, color='k', marker='*', edgecolors='none')

plt.show()