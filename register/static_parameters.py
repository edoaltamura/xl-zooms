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

