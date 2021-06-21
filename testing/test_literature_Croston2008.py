import sys
import numpy as np
from unyt import kpc
from matplotlib import pyplot as plt

sys.path.append("..")

from literature import Croston2008, Pratt2010
plt.style.use('../register/mnras.mplstyle')

pratt = Pratt2010()
names = pratt.Cluster_name
entropies = pratt.entropy_profiles / pratt.K500[:, None]
radii = pratt.radial_bins

print(entropies.shape)

lit = Croston2008(disable_cosmo_conversion=True)
lit.interpolate_r_r500(radii)
lit.compute_gas_mass()
lit.estimate_total_mass()
lit.compute_gas_fraction()

gas_fractions = np.empty_like(entropies)

# Display the catalogue data
for i, cluster in enumerate(lit.cluster_data):
    for name in names:
        if name == cluster['name']:
            gas_fractions[i] = cluster['f_g'] / lit.fb0

print(gas_fractions.shape)

# Display the catalogue data
for fg, k in zip(gas_fractions, entropies):
    plt.plot(fg, k, c='k', alpha=0.7, lw=0.3)
#
# # plt.axhline(self.fb0)
# plt.xlabel(r'$r/r_{500}$')
# y_label = r'$h_{70}^{-3/2}\ f_g (<R)$'
# plt.ylabel(y_label)
# plt.title(f"REXCESS sample - {lit.citation}")
# plt.xscale('log')
# plt.yscale('log')
plt.ylim(0, 2)
plt.xlim(0, 1)
plt.show()
plt.close()