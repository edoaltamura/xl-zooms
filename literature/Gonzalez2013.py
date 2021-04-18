import re
import os
import unyt
import numpy as np
from typing import List
from matplotlib import pyplot as plt

from .cosmology import Article, repository_dir

comment = (
    "Data assumes WMAP7 as fiducial model. "
    "The quoted stellar baryon fractions include a deprojection correction. "
    "A0478, A2029, and A2390 are not part of the main sample. "
    "These clusters were included only in the X-ray analysis to extend the baseline to higher mass, but "
    "they have no photometry equivalent to the other systems with which to measure the stellar mass. "
    "At the high mass end, however, the stellar component contributes "
    "a relatively small fraction of the total baryons. "
    "The luminosities include appropriate e + k corrections for each galaxy from GZZ07. "
    "The stellar masses are quoted as observed, with no deprojection correction applied"
)


class Gonzalez2013(Article):
    data_fields = ('ClusterID redshift TX2 Delta_TX2 L_BCG_ICL Delta_L_BCG_ICL LTotal Delta_LTotal r500 Delta_r500 M500'
                   'Delta_M500 M500_gas Delta_M500_gas M500_2D_star Delta_M500_2D_star M500_3D_star Delta_M500_3D_star'
                   'fgas Delta_fgas fstar Delta_fstar fbaryons Delta_fbaryons').split()

    def __init__(self, **cosmo_kwargs):
        super().__init__(
            citation="Gonzalez et al. (2013)",
            comment=comment,
            bibcode="2013ApJ...778...14G",
            hyperlink="https://ui.adsabs.harvard.edu/abs/2013ApJ...778...14G/abstract",
            **cosmo_kwargs
        )

        self.hconv = 0.702 / self.h
        self.process_data()

    def process_data(self):
        conversion_factors = [
            0,
            1,
            unyt.keV,
            unyt.keV,
            1.e12 * self.hconv ** 2 * unyt.Solar_Luminosity,
            1.e12 * self.hconv ** 2 * unyt.Solar_Luminosity,
            1.e12 * self.hconv ** 2 * unyt.Solar_Luminosity,
            1.e12 * self.hconv ** 2 * unyt.Solar_Luminosity,
            self.hconv * unyt.Mpc,
            self.hconv * unyt.Mpc,
            1.e14 * self.hconv * unyt.Solar_Mass,
            1.e14 * self.hconv * unyt.Solar_Mass,
            1.e13 * self.hconv ** 2 * unyt.Solar_Mass,
            1.e13 * self.hconv ** 2 * unyt.Solar_Mass,
            1.e13 * self.hconv ** 2 * unyt.Solar_Mass,
            1.e13 * self.hconv ** 2 * unyt.Solar_Mass,
            1.e13 * self.hconv ** 2 * unyt.Solar_Mass,
            1.e13 * self.hconv ** 2 * unyt.Solar_Mass,
            self.hconv * unyt.Dimensionless,
            self.hconv * unyt.Dimensionless,
            self.hconv * unyt.Dimensionless,
            self.hconv * unyt.Dimensionless,
            self.hconv * unyt.Dimensionless,
            self.hconv * unyt.Dimensionless,
        ]

        data = np.genfromtxt(os.path.join(repository_dir, 'gonzalez2013.dat'),
                             dtype=float,
                             invalid_raise=False,
                             missing_values='none',
                             usemask=False,
                             filling_values=np.nan).T

        for i, (field, conversion) in enumerate(zip(self.data_fields, conversion_factors)):
            setattr(self, field, data[i] * conversion)
