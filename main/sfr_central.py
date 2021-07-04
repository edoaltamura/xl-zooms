import os.path
import sys
import numpy as np
from velociraptor import load as vrload
from typing import Optional
from matplotlib import pyplot as plt


from unyt import (
    unyt_array,
    unyt_quantity,
    mh, G, mp, K, kb, cm, Solar_Mass, Mpc, dimensionless
)

from register import xlargs, find_files
snap, cat = find_files()

def set_snap_number(snap_number: int):
    old_snap_number = f"_{xlargs.snapshot_number:04d}"
    new_snap_number = f"_{snap_number:04d}"
    return snap.replace(old_snap_number, new_snap_number), cat.replace(old_snap_number, new_snap_number)


snaps_collection = np.arange(400, 1842, 200)
num_snaps = len(snaps_collection)
redshifts = np.empty(num_snaps)
sfr = np.empty(num_snaps)

for i, snap_number in enumerate(snaps_collection[::-1]):
    vr_data = vrload(set_snap_number(snap_number)[1])
    print(i, snap_number, f"Redshift {vr_data.z:.3f}")
    redshifts[i] = vr_data.z
    sfr[i] = vr_data.star_formation_rate.sfr_gas[0]

