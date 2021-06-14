import sys, os
import numpy as np
from matplotlib import pyplot as plt
from unyt import Solar_Mass

sys.path.append("..")
from literature import Sun2009, Pratt2010

plt.style.use('../register/mnras.mplstyle')

fig, axes = plt.subplots()
axes.set_xscale('log')
axes.set_yscale('log')
axes.axvline(0.15, color='k', linestyle='--', lw=0.5, zorder=0)
axes.set_ylabel(r'$K/K_{500}$')
axes.set_xlabel(r'$r/r_{500}$')
axes.set_ylim([5e-3, 20])
axes.set_xlim([5e-3, 2.5])

sun_observations = Sun2009()
r_r500, S_S500_50, S_S500_10, S_S500_90 = sun_observations.get_shortcut()

axes.fill_between(
    r_r500,
    S_S500_10,
    S_S500_90,
    color='grey', alpha=0.4, linewidth=0
)
axes.plot(r_r500, S_S500_50, c='grey', label=sun_observations.citation)

rexcess = Pratt2010()
bin_median, bin_perc16, bin_perc84 = rexcess.combine_entropy_profiles(
    m500_limits=(
        1e14 * Solar_Mass,
        5e14 * Solar_Mass
    ),
    k500_rescale=True
)

axes.fill_between(
    rexcess.radial_bins,
    bin_perc16,
    bin_perc84,
    color='lime',
    alpha=0.4,
    linewidth=0
)
axes.plot(rexcess.radial_bins, bin_median, c='lime', label=rexcess.citation)
axes.plot(np.array([0.01, 2.5]), 1.40 * np.array([0.01, 2.5]) ** 1.1, c='grey', ls='--', label='VKB (2005)')
plt.legend()
plt.show()
plt.close()
