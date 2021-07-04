import numpy as np
from velociraptor import load as vrload
from matplotlib import pyplot as plt
import unyt

import sys
sys.path.append("..")

from register import xlargs, find_files, set_mnras_stylesheet
set_mnras_stylesheet()
snap, cat = find_files()
sfr_output_units = unyt.msun / (unyt.year)# * unyt.Mpc ** 3)

def set_snap_number(snap_number: int):
    old_snap_number = f"_{xlargs.snapshot_number:04d}"
    new_snap_number = f"_{snap_number:04d}"
    return snap.replace(old_snap_number, new_snap_number), cat.replace(old_snap_number, new_snap_number)


snaps_collection = np.arange(200, 2500, 5)
num_snaps = len(snaps_collection)
redshifts = np.empty(num_snaps)
sfr = np.empty(num_snaps)

for i, snap_number in enumerate(snaps_collection[::-1]):
    vr_data = vrload(set_snap_number(snap_number)[1])
    print(i, snap_number, f"Redshift {vr_data.z:.3f}")
    # r200 = vr_data.radii.r_200crit[0]
    # volume = 4 / 3 * np.pi * r200 ** 3
    redshifts[i] = vr_data.z
    sfr[i] = (vr_data.star_formation_rate.sfr_gas[0]).to(sfr_output_units)

fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()
axes.set_yscale('log')
axes.set_xlabel('Redshift')
axes.set_ylabel(r"SFR $\dot{\rho}_*$ [M$_\odot$ yr$^{-1}$]")
axes.plot(redshifts, sfr, color='g', linewidth=0.5, alpha=1)

if not xlargs.quiet:
    plt.show()

plt.savefig('sfr_central.pdf')

plt.close()