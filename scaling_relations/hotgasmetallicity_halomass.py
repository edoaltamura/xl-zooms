# Plot scaling relations for EAGLE-XL tests
import sys
import os
import unyt
import numpy as np
from typing import Tuple
import h5py as h5
import swiftsimio as sw
import pandas as pd
import matplotlib.pyplot as plt

# Make the register backend visible to the script
sys.path.append("../zooms")
sys.path.append("../observational_data")

from register import zooms_register, Zoom, Tcut_halogas, name_list
import observational_data as obs
import scaling_utils as utils
import scaling_style as style

try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

cosmology = obs.Observations().cosmo_model
fbary = cosmology.Ob0 / cosmology.Om0  # Cosmic baryon fraction


def process_single_halo(
        path_to_snap: str,
        path_to_catalogue: str
) -> Tuple[unyt.unyt_quantity]:
    # Read in halo properties
    with h5.File(f'{path_to_catalogue}', 'r') as h5file:
        M500c = unyt.unyt_quantity(h5file['/SO_Mass_500_rhocrit'][0] * 1.e10, unyt.Solar_Mass)
        R500c = unyt.unyt_quantity(h5file['/SO_R_500_rhocrit'][0], unyt.Mpc)
        Thot500c = unyt.unyt_quantity(h5file['/SO_T_gas_highT_1.000000_times_500.000000_rhocrit'][0], unyt.K)
        Zhot500c = unyt.unyt_quantity(h5file['/SO_Zmet_gas_highT_1.000000_times_500.000000_rhocrit'][0], unyt.K)
        XPotMin = unyt.unyt_quantity(h5file['/Xcminpot'][0], unyt.Mpc)
        YPotMin = unyt.unyt_quantity(h5file['/Ycminpot'][0], unyt.Mpc)
        ZPotMin = unyt.unyt_quantity(h5file['/Zcminpot'][0], unyt.Mpc)

    # Read in gas particles to compute the core-excised temperature
    mask = sw.mask(f'{path_to_snap}', spatial_only=False)
    region = [[XPotMin - R500c, XPotMin + R500c],
              [YPotMin - R500c, YPotMin + R500c],
              [ZPotMin - R500c, ZPotMin + R500c]]
    mask.constrain_spatial(region)
    mask.constrain_mask(
        "gas", "temperatures",
        Tcut_halogas * mask.units.temperature,
        1.e12 * mask.units.temperature
    )
    data = sw.load(f'{path_to_snap}', mask=mask)
    posGas = data.gas.coordinates
    massGas = data.gas.masses
    mass_weighted_temperatures = data.gas.temperatures * data.gas.masses
    iron_fraction = data.gas.element_mass_fractions.iron * data.gas.masses

    # Select hot gas within sphere and without core
    deltaX = posGas[:, 0] - XPotMin
    deltaY = posGas[:, 1] - YPotMin
    deltaZ = posGas[:, 2] - ZPotMin
    deltaR = np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)

    index = np.where(deltaR < R500c)[0]
    iron_fraction_500c = np.sum(iron_fraction[index]) / np.sum(massGas[index])

    index = np.where((deltaR > 0.15 * R500c) & (deltaR < R500c))[0]
    Thot500c_nocore = np.sum(mass_weighted_temperatures[index]) / np.sum(massGas[index])

    return M500c, Thot500c, Thot500c_nocore, Zhot500c, iron_fraction_500c

@utils.set_scaling_relation_name(os.path.splitext(os.path.basename(__file__))[0])
@utils.set_output_names([
    'M_500crit',
    'Thot500c',
    'Thot500c_nocore'
])
def _process_single_halo(zoom: Zoom):
    return process_single_halo(zoom.snapshot_file, zoom.catalog_file)
