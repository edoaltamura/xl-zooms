import re
import os
import unyt
import numpy as np
from typing import List
from matplotlib import pyplot as plt

from .cosmology import Article, repository_dir

comment = (
    "Data in h_70 units."
    "Mstar−M500 relation",
    "a = 0.89 ± 0.14 and b = 12.44 ± 0.03",
    r"$\log \left(\frac{M_{\text {star }}}{\mathrm{M}_{\odot}}\right)=a \log \left(\frac{M_{500}}",
    r"{3 \times 10^{14} \mathrm{M}_{\odot}}\right)+b$",
    "star−M500 relation",
    "alpha = −0.11 ± 0.14 and beta = −2.04 ± 0.03",
    r"$\log f_{\text {star }}=\alpha \log \left(\frac{M_{500}}{3 \times 10^{14} "
    r"\mathrm{M}_{\odot}}\right)+\beta$"
)


class Budzynski2014(Article):
    M500_Bud = np.array([10 ** 13.7, 10 ** 15]) * unyt.Solar_Mass
    Mstar500_Bud = 10. ** (0.89 * np.log10(M500_Bud / 3.e14) + 12.44) * unyt.Solar_Mass
    Mstar500_Bud_a = (0.89, 0.14)
    Mstar500_Bud_b = (12.44, 0.03)

    def __init__(self, **cosmo_kwargs):
        super().__init__(
            citation="Budzynski et al. (2014)",
            comment=comment,
            bibcode="2014MNRAS.437.1362B",
            hyperlink="https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.1362B/abstract",
            **cosmo_kwargs
        )
        self.hconv = 0.71 / self.h

        self.M500 = self.M500_Bud * self.hconv
        self.Mstar500 = self.Mstar500_Bud * self.hconv ** 2.5
        self.fit_line_uncertainty_weights()
        self.M500_trials *= self.hconv * unyt.Solar_Mass
        self.Mstar_trials_upper *= self.hconv ** 2.5 * unyt.Solar_Mass
        self.Mstar_trials_median *= self.hconv ** 2.5 * unyt.Solar_Mass
        self.Mstar_trials_lower *= self.hconv ** 2.5 * unyt.Solar_Mass

    def fit_line_uncertainty_weights(self):
        np.random.seed(0)

        # Generate random samples for the Mstar relation
        M500_trials = np.logspace(13.7, 15., 20)
        a_trials = np.random.normal(self.Mstar500_Bud_a[0], self.Mstar500_Bud_a[1], 20)
        b_trials = np.random.normal(self.Mstar500_Bud_b[0], self.Mstar500_Bud_b[1], 20)
        Mstar_trials = np.empty(0)

        for a in a_trials:
            for b in b_trials:
                Mstar_trial = 10 ** (a * np.log10(M500_trials / 3.e14) + b)
                Mstar_trials = np.append(Mstar_trials, Mstar_trial)

        Mstar_trials = Mstar_trials.reshape(-1, M500_trials.size)
        self.M500_trials = M500_trials
        self.Mstar_trials_upper = np.percentile(Mstar_trials, 16, axis=0)
        self.Mstar_trials_median = np.percentile(Mstar_trials, 50, axis=0)
        self.Mstar_trials_lower = np.percentile(Mstar_trials, 84, axis=0)
