import re
import os
import unyt
import numpy as np
from typing import List
from matplotlib import pyplot as plt

from .cosmology import Article, repository_dir

comment = (
    "REXCESS sample."
)


class Bohringer2007(Article):

    def __init__(self, **cosmo_kwargs):
        super().__init__(
            citation="Böhringer et al. (2007)",
            comment=comment,
            bibcode="2007A&A...469..363B",
            hyperlink="https://ui.adsabs.harvard.edu/abs/2007A%26A...469..363B/abstract",
            **cosmo_kwargs
        )
        self.bins = [(0.407, 0.55, 0.0555, 0.06215),
                     (0.55, 0.78, 0.0794, 0.0877),
                     (0.78, 1.13, 0.0920, 0.1037),
                     (1.13, 1.71, 0.1077, 0.12105),
                     (1.71, 2.88, 0.1122, 0.1248),
                     (2.88, 4.10, 0.1224, 0.15215),
                     (4.10, 5.90, 0.1337, 0.16875),
                     (5.90, 11.9, 0.1423, 0.1719),
                     (11.9, 20.0, 0.1623, 0.19925)]

    def draw_LX_bounds(self, ax: plt.Axes, redshifts_on: bool = True):
        self.hconv = 0.70 / self.h

        ax.axhspan(
            self.bins[0][0] * 1e44 * self.hconv ** 2,
            self.bins[-1][1] * 1e44 * self.hconv ** 2,
            facecolor='lime',
            linewidth=0,
            alpha=0.2
        )
        ax.axhline(
            self.bins[0][0] * 1e44 * self.hconv ** 2,
            color='lime',
            linewidth=1,
            alpha=0.1
        )

        for i, (
                luminosity_min,
                luminosity_max,
                redshift_min,
                redshift_max
        ) in enumerate(self.bins):

            ax.axhline(
                luminosity_max * 1e44 * self.hconv ** 2,
                color='lime',
                linewidth=1,
                alpha=0.1
            )

            # Print redshift bounds once every 2 bins to avoid clutter.
            if i % 2 == 0 and redshifts_on:
                ax.text(
                    10 ** ax.get_xlim()[0],
                    10 ** (0.5 * np.log10(luminosity_min * luminosity_max)) * 1e44 * self.hconv ** 2,
                    f"$z$ = {redshift_min:.3f} - {redshift_max:.3f}",
                    horizontalalignment='left',
                    verticalalignment='center',
                    color='k',
                    alpha=0.3
                )

    def quick_display(self):
        fig, ax = plt.subplots()
        self.draw_LX_bounds(ax)
        plt.title(f"REXCESS sample - {self.citation}")
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.show()
        plt.close()
