import re
import os
import unyt
import numpy as np
from typing import List
from matplotlib import pyplot as plt

from .cosmology import Article, repository_dir

comment = (
    "Pressure and temperature profiles for the REXCESS sample "
    "Pressure profiles are GNFW fits"
    "Corrected for the original cosmology which had h=0.7"
)


class Arnaud2010(Article):

    def __init__(self, **cosmo_kwargs):
        super().__init__(
            citation="Arnaud et al. (2010)",
            comment=comment,
            bibcode="2010A&A...517A..92A",
            hyperlink="https://ui.adsabs.harvard.edu/abs/2010A%26A...517A..92A/abstract",
            **cosmo_kwargs
        )

        # Read the data
        P500, P0, c500, alpha, gamma = np.loadtxt(f'{repository_dir}/Arnaud2010.dat').T
        P500 *= 1e-3
        number_objects = len(P0)

        # Build profiles from fit parameters
        r_r500 = np.logspace(np.log10(0.03), 0, 30)
        self.dimensionless_pressure_profiles = np.zeros((number_objects, len(r_r500)), dtype=np.float64)
        for i in range(number_objects):
            self.dimensionless_pressure_profiles = self.gnfw(
                r_r500, P0[i], c500[i], alpha=alpha[i], beta=5.49, gamma=gamma[i]
            )
        print(self.dimensionless_pressure_profiles)

        self.dimensionless_pressure_profiles_median = np.median(self.dimensionless_pressure_profiles, axis=1)
        self.dimensionless_pressure_profiles_perc84 = np.percentile(self.dimensionless_pressure_profiles, 84, axis=1)
        self.dimensionless_pressure_profiles_perc16 = np.percentile(self.dimensionless_pressure_profiles, 16, axis=1)

        # Assign units
        self.P500 = unyt.unyt_array(P500, 'keV/cm**3')
        self.scaled_radius = unyt.unyt_array(r_r500)
        self.dimensionless_pressure_profiles_median = unyt.unyt_array(self.dimensionless_pressure_profiles_median)
        self.dimensionless_pressure_profiles_perc84 = unyt.unyt_array(self.dimensionless_pressure_profiles_perc84)
        self.dimensionless_pressure_profiles_perc16 = unyt.unyt_array(self.dimensionless_pressure_profiles_perc16)

    @staticmethod
    def gnfw(x, p0: float, c500: float, alpha: float, beta: float, gamma: float):
        """
        In this paper, beta is fixed at = 5.49
        :param x: np.array
            Radial distance scaled by R_{500}: r/r500
        :param p0: normalisation constant
        :param c500: NFW concentration
        :param alpha: intermediate slope
        :param beta: outer slope
        :param gamma: central slope
        :return:
        """
        c500x = x * c500
        exponent = (beta - gamma) / alpha
        return p0 / (c500x ** gamma * (1 + c500x ** alpha) ** exponent)
