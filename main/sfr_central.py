import numpy as np
from velociraptor import load as vrload
from matplotlib import pyplot as plt
import unyt

import sys

sys.path.append("..")

from register import xlargs, find_files, set_mnras_stylesheet

set_mnras_stylesheet()
snap, cat = find_files()
sfr_output_units = unyt.msun / (unyt.year)  # * unyt.Mpc ** 3)


def set_snap_number(snap_number: int):
    old_snap_number = f"_{xlargs.snapshot_number:04d}"
    new_snap_number = f"_{snap_number:04d}"
    return snap.replace(old_snap_number, new_snap_number), cat.replace(old_snap_number, new_snap_number)


snaps_collection = np.arange(200, 2500, 50)
num_snaps = len(snaps_collection)
redshifts = np.empty(num_snaps)
sfr = np.empty(num_snaps)

for i, snap_number in enumerate(snaps_collection[::-1]):

    try:
        vr_data = vrload(set_snap_number(snap_number)[1])
        print(i, snap_number, f"Redshift {vr_data.z:.3f}")
        # r200 = vr_data.radii.r_200crit[0]
        # volume = 4 / 3 * np.pi * r200 ** 3
        redshifts[i] = vr_data.z
        sfr[i] = (vr_data.star_formation_rate.sfr_gas[0]).to(sfr_output_units)
    except:
        redshifts[i] = np.nan
        sfr[i] = np.nan

fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()
axes.set_yscale('log')
axes.set_xscale('log')
axes.set_xlabel('Redshift')
axes.set_ylabel(r"SFR $\dot{\rho}_*$ [M$_\odot$ yr$^{-1}$]")

scale_factors = 1 / (redshifts + 1)
axes.plot(scale_factors, sfr, color='g', linewidth=0.5, alpha=1)

redshift_ticks = np.array([0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0])
redshift_labels = [
    "$0$",
    "$0.2$",
    "$0.5$",
    "$1$",
    "$2$",
    "$3$",
    "$5$",
    "$10$",
    "$20$",
    "$50$",
    "$100$",
]
a_ticks = 1.0 / (redshift_ticks + 1.0)

axes.set_xticks(a_ticks)
axes.set_xticklabels(redshift_labels)
axes.tick_params(axis="x", which="minor", bottom=False)
# axes.set_xlim(1.02, 0.07)
# axes.set_ylim(1.8e-4, 1.7)

if not xlargs.quiet:
    plt.show()

plt.savefig('sfr_central.pdf')

plt.close()
