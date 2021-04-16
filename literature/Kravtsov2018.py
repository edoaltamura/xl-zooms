import re
import os
import unyt
import numpy as np
from typing import List
from matplotlib import pyplot as plt

from .cosmology import Article, repository_dir

comment = (
    "Stellar Mass—Halo Mass Relation and Star Formation Efficiency in High-Mass Halos. "
    "Data in h_70 units."
)


class Kravtsov2018(Article):

    def __init__(self, **cosmo_kwargs):
        super().__init__(
            citation="Kravtsov et al. (2018)",
            comment=comment,
            bibcode="2018AstL...44....8K",
            hyperlink="https://ui.adsabs.harvard.edu/abs/2018AstL...44....8K/abstract",
            **cosmo_kwargs
        )

        M500_Kra = np.array(
            [15.60, 10.30, 7.00, 5.34, 2.35, 1.86, 1.34, 0.46, 0.47]
        ) * 10. * 1.e13 * unyt.Solar_Mass
        Mstar500_Kra = np.array(
            [15.34, 12.35, 8.34, 5.48, 2.68, 3.48, 2.86, 1.88, 1.85]
        ) * 0.1 * 1.e13 * unyt.Solar_Mass

        self.hconv = 0.70 / self.h
        self.M500 = M500_Kra * self.hconv
        self.Mstar500 = Mstar500_Kra * self.hconv ** 2.5
