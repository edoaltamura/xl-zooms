import re
import os
import unyt
import numpy as np
from typing import List
from matplotlib import pyplot as plt

from .cosmology import Article, repository_dir

comment = (
    "The XXL Survey. XIII. Baryon content of the bright cluster sample"
    "Based on observations obtained with XMM-Newton. "
    "Corrected for the original cosmology which had h=0.7"
)


class Eckert2016(Article):

    def __init__(self, **cosmo_kwargs):
        super().__init__(
            citation="Eckert et al. (2016)",
            comment=comment,
            bibcode="2016A&A...592A..12E",
            hyperlink="https://ui.adsabs.harvard.edu/abs/2016A%26A...592A..12E/abstract",
            **cosmo_kwargs
        )

        self.hconv = 0.70 / self.h

        # Read the data
        raw = np.loadtxt(f'{repository_dir}/Eckert2016.dat')
        self.M_500 = unyt.unyt_array(self.hconv * 10 ** raw[:, 0], units="Msun")
        self.fb_500 = unyt.unyt_array(raw[:, 1] * self.hconv ** 2.5, units="dimensionless")
        self.M_500gas = self.M_500 * self.fb_500

        self.M_500_fit = np.logspace(raw[:, 0].min(), raw[:, 0].max(), 60)
        self.fgas_500_fit = 0.055 * (self.M_500_fit * self.hconv / 1e14) ** 0.21 * self.hconv ** (3 / 2)
        self.M_500gas_fit = self.M_500_fit * self.fgas_500_fit

        # self.fit_line_uncertainty_weights()

    def fit_line_uncertainty_weights(self):
        np.random.seed(0)
        # Generate random samples for the fgas relation
        M500_trials = self.M_500_fit
        a_trials = np.random.normal(0.055, 0.007, 30)
        b_trials = np.random.normal(0.21, 0.11, 40)
        fgas_trials = np.empty(0)
        for a in a_trials:
            for b in b_trials:
                Mstar_trial = a * (M500_trials * self.hconv / 1e14) ** b * self.hconv ** (3 / 2)
                fgas_trials = np.append(fgas_trials, Mstar_trial)
        fgas_trials = fgas_trials.reshape(-1, M500_trials.size)
        self.M500_trials = M500_trials
        self.fgas_trials_trials_upper = np.percentile(fgas_trials, 16, axis=0)
        self.fgas_trials_trials_median = np.percentile(fgas_trials, 50, axis=0)
        self.fgas_trials_trials_lower = np.percentile(fgas_trials, 84, axis=0)

    def quick_display(self):
        self.fit_line_uncertainty_weights()
        # Display the catalogue data
        plt.scatter(self.M_500, self.fb_500, c='orange', s=2, label='baryon fractions')
        plt.plot(self.M_500_fit, self.fgas_500_fit, label='best fit median')
        plt.fill_between(
            self.M500_trials,
            self.fgas_trials_trials_lower,
            self.fgas_trials_trials_upper,
            color='aqua', alpha=0.85, linewidth=0
        )
        plt.plot(self.M500_trials, self.fgas_trials_trials_median, label='resampled median')
        # plt.plot([self.M_500.min(), self.M_500.max()], [0.14, 0.14], linestyle='--')
        plt.ylabel(r'$f_{\rm gas}\ (R_{500})$')
        plt.xlabel(r'$M_{500}$ [M$_\odot$]')
        plt.title(self.citation)
        plt.xscale('log')
        plt.legend()
        plt.show()
        plt.close()
