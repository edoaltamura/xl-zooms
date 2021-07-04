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

def set_snap_number(snap: str, cat: str, snap_number: int):
    old_snap_number = f"_{xlargs.snapshot_number:04d}"
    new_snap_number = f"_{snap_number:04d}"
    return snap.replace(old_snap_number, new_snap_number), cat.replace(old_snap_number, new_snap_number)

def setup_data(snap_number):
    vr_data = vrload(path_to_catalogue)
    return.star_formation_rate.sfr_gas