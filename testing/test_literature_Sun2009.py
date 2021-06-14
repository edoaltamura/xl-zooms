import sys
import numpy as np
from unyt import kpc
from matplotlib import pyplot as plt

sys.path.append("..")

from literature import Sun2009, Pratt2010
plt.style.use('../register/mnras.mplstyle')
lit = Sun2009(reduced_table=True, disable_cosmo_conversion=False)
from matplotlib import pyplot as plt

x = [
    np.median(30 * kpc / lit.R_500),
    np.median(0.15),
    np.median(lit.R_2500 / lit.R_500),
    np.median(lit.R_1500 / lit.R_500),
    np.median(lit.R_1000 / lit.R_500),
    np.median(1),
]
y = [
    np.median(lit.K_500 / lit.K_500_adi),
    np.median(lit.K_1000 / lit.K_500_adi),
    np.median(lit.K_1500 / lit.K_500_adi),
    np.median(lit.K_2500 / lit.K_500_adi),
    np.median(lit.K_0p15r500 / lit.K_500_adi),
    np.median(lit.K_30kpc / lit.K_500_adi),
][::-1]




plt.loglog()
plt.scatter(x, y, label='Edo')
for i, value in enumerate(y):
    plt.annotate(f"{value.v:.2f}", (x[i], y[i]))

r_r500, S_S500_50, S_S500_10, S_S500_90 = lit.get_shortcut()

plt.fill_between(
    r_r500,
    S_S500_10,
    S_S500_90,
    color='grey', alpha=0.5, linewidth=0
)
plt.plot(r_r500, S_S500_50, c='k', label='Jaime')

rexcess = Pratt2010()
bin_median, bin_perc16, bin_perc84 = rexcess.combine_entropy_profiles(
    k500_rescale=True
)
print(bin_median[0], bin_median[-1])

plt.fill_between(
    rexcess.radial_bins,
    bin_perc16,
    bin_perc84,
    color='lime',
    alpha=0.4,
    linewidth=0
)
plt.plot(rexcess.radial_bins, bin_median, c='lime', label=rexcess.citation)

plt.xlabel(r'$r/r_{500}$')
plt.ylabel(r'$K/K_{500adi}$')
plt.xlim([0.01, 1])
plt.legend()
plt.show()

# lit.filter_by('M_500', 8e13, 3e14)
# lit.overlay_points(None, 'T_2500', 'K_30kpc')
# lit.overlay_entropy_profiles(k_units='K500adi')
