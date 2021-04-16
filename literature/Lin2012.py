import re
import os
import unyt
import numpy as np
from typing import List
from matplotlib import pyplot as plt

from .cosmology import Article, repository_dir

comment = (
    "Halo mass - Gas fraction relation from Chandra-observed clusters."
    "Ionized gas out of 94 clusters combining Chandra, WISE and 2MASS. "
    "Data was corrected for the simulation's cosmology."
)


class Lin2012(Article):

    def __init__(self, **cosmo_kwargs):
        super().__init__(
            citation="Lin et al. (2012; $z<0.25$ only)",
            comment=comment,
            bibcode="2012ApJ...745L...3L",
            hyperlink="https://ui.adsabs.harvard.edu/abs/2012ApJ...745L...3L/abstract",
            **cosmo_kwargs
        )

        self.hconv = 0.71 / self.h

        # Read the data
        raw = np.loadtxt(f'{repository_dir}/Lin2012.dat')
        M_500 = unyt.unyt_array(self.hconv * 10 ** raw[:, 0], units="Msun")
        M_500_error = unyt.unyt_array(self.hconv * raw[:, 1], units="Msun")
        M_500_gas = unyt.unyt_array(self.hconv * 10 ** raw[:, 2], units="Msun")
        M_500_gas_error = unyt.unyt_array(self.hconv * raw[:, 3], units="Msun")
        z = raw[:, 6]

        # Compute the gas fractions
        fb_500 = (M_500_gas / M_500) * self.hconv ** 2.5
        fb_500_error = fb_500 * ((M_500_error / M_500) + (M_500_gas_error / M_500_gas))

        # Normalise by the cosmic mean
        # fb_500 = fb_500 / (Omega_b / Omega_m)
        # fb_500_error = fb_500_error / (Omega_b / Omega_m)

        # Select only the low-z data
        self.M_500 = M_500[z < 0.25]
        self.fb_500 = fb_500[z < 0.25]
        self.M_500gas = M_500_gas[z < 0.25]

        self.M_500_error = M_500_error[z < 0.25]
        self.fb_500_error = fb_500_error[z < 0.25]
        self.M_500gas_error = M_500_gas_error[z < 0.25]
