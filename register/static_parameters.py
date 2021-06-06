# -*- coding: utf-8 -*-
"""Definition of boot-up parameters

This file contains definitions of static variables used across the analysis
pipeline. Changing their value in this file should apply changes in all the
routines that call them.
"""
import os

Tcut_halogas = 1.e5  # Kelvin
default_output_directory: str = "/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"

cooling_table: str = "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/UV_dust1_CR1_G1_shield1.hdf5"

if not os.path.isdir(default_output_directory):
    raise NotADirectoryError("The default directory does not exist")

matplotlib_stylesheet = os.path.join(
    os.path.dirname(__file__),
    'mnras.mplstyle'
)

# Constants
mean_molecular_weight = 0.5954  # mean atomic weight for an ionized gas with primordial (X = 0.76, Z = 0) composition
mean_atomic_weight_per_free_electron = 1.14
primordial_hydrogen_mass_fraction = 0.76
solar_metallicity = 0.0133714
gamma = 5 / 3

# Aliases
mu = mean_molecular_weight
mu_e = mean_atomic_weight_per_free_electron
f_H = primordial_hydrogen_mass_fraction
Zsun = solar_metallicity


