import re
import os
import unyt
import numpy as np
from typing import List
from matplotlib import pyplot as plt

from .cosmology import Article, repository_dir

comment = (
    "Halo mass - Gas fraction relation from 13 low-redshift Chandra-observed relaxed clusters."
    "Gas and total mass profiles for 13 low-redshift, relaxed clusters spanning "
    "a temperature range 0.7-9 keV, derived from all available Chandra data of "
    "sufficient quality. Data was corrected for the simulation's cosmology."
)


class Vikhlinin2006(Article):

    def __init__(self, **cosmo_kwargs):
        super().__init__(
            citation="Vikhlinin et al. (2006)",
            comment=comment,
            bibcode="2006ApJ...640..691V",
            hyperlink="https://ui.adsabs.harvard.edu/abs/2006ApJ...640..691V/abstract",
            **cosmo_kwargs
        )

        self.hconv = 0.72 / self.h

        # Read the data
        raw = np.loadtxt(f'{repository_dir}/Vikhlinin2006.dat')
        self.M_500 = unyt.unyt_array((10 ** 14) * self.hconv * raw[:, 1], units="Msun")
        self.error_M_500 = unyt.unyt_array((10 ** 14) * self.hconv * raw[:, 2], units="Msun")
        self.fb_500 = unyt.unyt_array(self.hconv ** 1.5 * raw[:, 3], units="dimensionless")
        self.error_fb_500 = unyt.unyt_array(self.hconv ** 1.5 * raw[:, 4], units="dimensionless")
        self.M_500gas = self.M_500 * self.fb_500
        self.error_M_500gas = self.M_500gas * (self.error_M_500 / self.M_500 + self.error_fb_500 / self.fb_500)
