from merge_catalogues import catalogue, select_runs

from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.legend_handler import HandlerLine2D
import unyt
import numpy as np
import swiftsimio as sw
import velociraptor as vr
from typing import Tuple
import matplotlib.patheffects as path_effects

try:
    plt.style.use("../register/mnras.mplstyle")
except:
    pass


fig = plt.figure(figsize=(9, 5))
gs = fig.add_gridspec(2, 3, hspace=0.05, wspace=0.3)
axes = gs.subplots(sharex=True, sharey=False)

shadow = dict(path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])

for ax in axes.flat:
    ax.loglog()
    ax.axvline(x=1, color='k', linestyle='--')
    ax.set_xlim([1e-3, 2])


# 'Run_name', 'm_gas', 'f_gas', 'T500', 'T2500', 'T500_nocore',
#        'T2500_nocore', 'r2500', 'r1000', 'r500', 'r200', 'm2500', 'm1000',
#        'm500', 'm200', 'm_star500c', 'm_star30kpc', 'm_star', 'f_star',
#        'k30kpc', 'k2500', 'k1500', 'k1000', 'k500', 'k200', 'Ekin', 'Etherm',
#        'r_1500', 'm_1500', 'LX500', 'LX2500', 'LX500_nocore', 'LX2500_nocore'

axes[0, 0].scatter(catalogue['T500_nocore'], catalogue['k500'])
axes[0, 1].scatter(catalogue['T500_nocore'], catalogue['k1000'])
axes[0, 2].scatter(catalogue['T500_nocore'], catalogue['k1500'])
axes[1, 0].scatter(catalogue['T2500_nocore'], catalogue['k2500'])
axes[1, 1].scatter(catalogue['T2500_nocore'], catalogue['k500'])
axes[1, 2].scatter(catalogue['T2500_nocore'], catalogue['k30kpc'])

plt.show()