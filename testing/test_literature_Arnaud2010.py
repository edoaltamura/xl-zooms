import sys
import numpy as np
from unyt import kpc
from matplotlib import pyplot as plt

sys.path.append("..")

from literature import Arnaud2010
plt.style.use('../register/mnras.mplstyle')

plt.ylabel(r'Pressure $P/P_{500}$ $(r/r_{500})^3$')
plt.xlabel(r'$r/r_{500}$')

arnaud = Arnaud2010()
median = arnaud.dimensionless_temperature_profiles_median
perc16 = arnaud.dimensionless_temperature_profiles_perc16
perc84 = arnaud.dimensionless_temperature_profiles_perc84
plt.errorbar(
    arnaud.scaled_radius,
    median,
    yerr=(
        median - perc16,
        perc84 - median
    ),
    fmt='s',
    color='grey',
    ecolor='lightgray',
    elinewidth=0.5,
    capsize=0,
    markersize=1,
    label=arnaud.citation,
    zorder=0
)
plt.legend()

plt.xscale('log')
# plt.yscale('log')
plt.ylim(0, 3)
plt.show()
plt.close()
