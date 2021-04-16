import re
import os
import unyt
import numpy as np
from typing import List
from matplotlib import pyplot as plt

from .cosmology import Article, repository_dir

comment = (
    "Scaling Properties of a Complete X-ray Selected Galaxy Group Sample"
    "23 Galaxy groups observed with XMM-Newton"
    "Corrected for the original cosmology which had h=0.7"
)


class Lovisari2015(Article):

    def __init__(self, **cosmo_kwargs):
        super().__init__(
            citation="Lovisari et al. (2015)",
            comment=comment,
            bibcode="2015A&A...573A.118L",
            hyperlink="https://ui.adsabs.harvard.edu/abs/2015A%26A...573A.118L/abstract",
            **cosmo_kwargs
        )

        # Read the data
        raw = np.loadtxt(f'{repository_dir}/Lovisari2015.dat')
        self.M_500 = unyt.unyt_array((0.70 / self.cosmo_model.h) * 10 ** raw[:, 0], units="Msun")
        self.fb_500 = unyt.unyt_array(raw[:, 1] * (0.70 / self.cosmo_model.h) ** 2.5, units="dimensionless")
        self.M_gas500 = self.M_500 * self.fb_500
