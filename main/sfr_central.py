import numpy as np
from velociraptor import load as vrload
from matplotlib import pyplot as plt
import unyt

import sys

sys.path.append("..")

from register import xlargs, find_files, set_mnras_stylesheet, delete_last_line

set_mnras_stylesheet()
snap, cat = find_files()
sfr_output_units = unyt.msun / unyt.Gyr


def set_snap_number(snap_number: int):
    old_snap_number = f"_{xlargs.snapshot_number:04d}"
    new_snap_number = f"_{snap_number:04d}"
    return snap.replace(old_snap_number, new_snap_number), cat.replace(old_snap_number, new_snap_number)


def set_resolution(old_path: str, resolution: str):
    old_resolution = '+1res' if '+1res' in old_path else '-8res'
    return old_path.replace(old_resolution, resolution)


def set_vr_number(old_path: str, vr_number: str):
    old_vr_number = old_path.split('_')[1][2:]
    return old_path.replace(old_vr_number, vr_number)


fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()
axes.set_yscale('log')
axes.set_xscale('log')
axes.set_xlabel('Redshift')
# axes.set_ylabel(r"Specific SFR = $\dot{M}_* / M_*$(100 kpc) [Gyr$^{-1}$]")
# axes.set_ylabel(r"SFR = $\dot{M}_*$ [M$_\odot$ yr$^{-1}$]")
# axes.set_ylabel(r"M$_*$(100 kpc) [M$_\odot$]")
axes.set_ylabel(r"M$_{\rm BH}$(10 kpc) [M$_\odot$]")


for vr_number in ['37', '139', '485', '680', '813', '1236', '2414', '2915']:
    for resolution in ['-8res', '+1res']:

        # snaps_collection = np.arange(1, 2522, 20)
        snaps_collection = np.arange(36)
        num_snaps = len(snaps_collection)
        redshifts = np.empty(num_snaps)
        sfr = np.empty(num_snaps)
        mass_bcg = np.empty(num_snaps)
        mass_bh = np.empty(num_snaps)


        for i, snap_number in enumerate(snaps_collection[::-1]):

            catalog_path = set_snap_number(snap_number)[1]
            catalog_path = set_vr_number(catalog_path, vr_number)
            catalog_path = set_resolution(catalog_path, resolution)

            try:
                vr_data = vrload(catalog_path, disregard_units=True)

                if snap_number == snaps_collection[-1]:
                    print('m500', vr_data.spherical_overdensities.mass_500_rhocrit[0])

                print(f"Snap number {snap_number} Redshift {vr_data.z:.3f} VR number {vr_number} Resolution {resolution}")

                redshifts[i] = vr_data.z
                mass_bcg[i] = vr_data.apertures.mass_star_100_kpc[0].to(unyt.msun)
                mass_bh[i] = vr_data.apertures.mass_bh_10_kpc[0].to(unyt.msun)
                sfr[i] = (vr_data.apertures.sfr_gas_100_kpc[0]).to(sfr_output_units)

                delete_last_line()
            except:
                redshifts[i] = np.nan
                sfr[i] = np.nan

        scale_factors = 1 / (redshifts + 1)
        axes.plot(scale_factors, mass_bh, color='g' if resolution is '-8res' else 'r', linewidth=0.5, alpha=1)


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
axes.set_xlim(1.02, 0.07)
# axes.set_ylim(1.8e-4, 1.7)

import matplotlib.patches as mpatches
axes.legend(
    handles=[
        mpatches.Patch(color='g', label='-8res'),
        mpatches.Patch(color='r', label='EAGLE res')
    ]
)

if not xlargs.quiet:
    plt.show()

plt.savefig('sfr_central.pdf')

plt.close()
