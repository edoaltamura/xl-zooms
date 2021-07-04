import os.path
import sys
import numpy as np
from swiftsimio import cosmo_array
from typing import Optional
from matplotlib import pyplot as plt
import numba
from multiprocessing import cpu_count

numba.config.NUMBA_NUM_THREADS = cpu_count()

from unyt import (
    unyt_array,
    unyt_quantity,
    mh, G, mp, K, kb, cm, Solar_Mass, Mpc, dimensionless
)

from register import (
    Zoom, Tcut_halogas, default_output_directory, xlargs,
    set_mnras_stylesheet,
    mean_molecular_weight,
    mean_atomic_weight_per_free_electron,
    primordial_hydrogen_mass_fraction,
    solar_metallicity,
    gamma,
)
from .halo_property import HaloProperty, histogram_unyt
from .spherical_overdensities import SODelta500
from hydrostatic_estimates import HydrostaticEstimator
from literature import Cosmology

class BHEnergyInjection(HaloProperty):

    def __init__(self):
        super().__init__()

    def setup_data(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None
    ):
        sw_data, vr_data = self.get_handles_from_zoom(
            zoom_obj,
            path_to_snap,
            path_to_catalogue,
            mask_radius_r500=0.5
        )
        self.fb = Cosmology().fb0
        self.critical_density = unyt_quantity(
            sw_data.metadata.cosmology.critical_density(sw_data.metadata.z).value, 'g/cm**3'
        ).to('Msun/Mpc**3')
        self.z = sw_data.metadata.z

        try:
            m500 = vr_data.spherical_overdensities.mass_500_rhocrit[0].to('Msun')
            r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc')
        except AttributeError as err:
            print(f'[{self.__class__.__name__}] {err}')

            spherical_overdensity = SODelta500(
                path_to_snap=path_to_snap,
                path_to_catalogue=path_to_catalogue,
            )
            m500 = spherical_overdensity.get_m500()
            r500 = spherical_overdensity.get_r500()

        if xlargs.mass_estimator == 'hse':
            true_hse = HydrostaticEstimator(
                path_to_catalogue=path_to_catalogue,
                path_to_snap=path_to_snap,
                profile_type='true',
                diagnostics_on=False
            )
            true_hse.interpolate_hse(density_contrast=500.)
            r500 = true_hse.r500hse
            m500 = true_hse.m500hse

        sw_data.black_holes.radial_distances.convert_to_physical()
        sw_data.black_holes.subgrid_masses.convert_to_physical()
        sw_data.black_holes.agntotal_injected_energies.convert_to_physical()