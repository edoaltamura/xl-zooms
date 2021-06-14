import re
import os
from unyt import (
    unyt_quantity, unyt_array,
    keV, cm, dimensionless, Solar_Mass, erg, s, kpc
)
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
import itertools

from .cosmology import Article, repository_dir

comment = (
    "REXCESS sample."
)


class Pratt2010(Article):

    def __init__(self, disable_cosmo_conversion: bool = False, **cosmo_kwargs):
        super().__init__(
            citation="Pratt et al. (2010)",
            comment=comment,
            bibcode="2010A&A...511A..85P",
            hyperlink='https://ui.adsabs.harvard.edu/abs/2010A%26A...511A..85P/abstract',
            **cosmo_kwargs
        )

        self.hconv = 0.70 / self.h
        if disable_cosmo_conversion:
            self.hconv = 1.

        self.process_properties()
        self.process_entropy_profiles()

    def process_properties(self):
        field_names = (
            'Cluster_name z '
            'kT Delta_hi_kT Delta_lo_kT '
            'M500 Delta_hi_M500 Delta_lo_M500 '
            'K0p1R200 Delta_K0p1R200 '
            'KR2500 Delta_KR2500 '
            'KR1000 Delta_KR1000 '
            'KR500 Delta_KR500 '
            'kT_R500 Delta_hi_kT_R500 Delta_lo_kT_R500 '
            'LX_R500 Delta_hi_LX_R500 Delta_lo_LX_R500 '
            'kT_0p15_1R500 Delta_hi_kT_0p15_1R500 Delta_lo_kT_0p15_1R500 '
            'LX_0p15_1R500 Delta_hi_LX_0p15_1R500 Delta_lo_LX_0p15_1R500 '
            'kT_0p15_0p75R500 Delta_hi_kT_0p15_0p75R500 Delta_lo_kT_0p15_0p75R500 '
            'YX_R500 Delta_hi_YX_R500 Delta_lo_YX_R500 '
            'R500 Rdet').split()

        data = []

        with open(f'{repository_dir}/pratt2010_properties.dat') as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith('#') and not line.isspace():
                    line_data = line.split()
                    for i, element_data in enumerate(line_data):
                        if element_data.strip() == 'none':
                            # If no data, replace with Nan
                            line_data[i] = np.nan
                        elif re.search('[a-df-zA-Z]', element_data):
                            # If contains letters, remove white spaces
                            line_data[i] = element_data.strip()
                        else:
                            line_data[i] = float(element_data.strip())
                    data.append(line_data)

        data = list(map(list, itertools.zip_longest(*data, fillvalue=None)))
        for i, field in enumerate(data):
            data[i] = np.array(field)

        conversion_factors = [
            None,
            1,
            keV,
            keV,
            keV,
            1.e14 / 0.7 * self.hconv * Solar_Mass,
            1.e14 / 0.7 * self.hconv * Solar_Mass,
            1.e14 / 0.7 * self.hconv * Solar_Mass,
            keV * cm ** 2,
            keV * cm ** 2,
            keV * cm ** 2,
            keV * cm ** 2,
            keV * cm ** 2,
            keV * cm ** 2,
            keV * cm ** 2,
            keV * cm ** 2,
            keV,
            keV,
            keV,
            1e44 * self.hconv ** 2 * erg / s,
            1e44 * self.hconv ** 2 * erg / s,
            1e44 * self.hconv ** 2 * erg / s,
            keV,
            keV,
            keV,
            1e44 * self.hconv ** 2 * erg / s,
            1e44 * self.hconv ** 2 * erg / s,
            1e44 * self.hconv ** 2 * erg / s,
            keV,
            keV,
            keV,
            1e13 * self.hconv * keV * Solar_Mass,
            1e13 * self.hconv * keV * Solar_Mass,
            1e13 * self.hconv * keV * Solar_Mass,
            kpc,
            dimensionless
        ]

        for i, (field, conversion) in enumerate(zip(field_names, conversion_factors)):
            if isinstance(data[i][0], str):
                setattr(self, field, data[i])
            else:
                setattr(self, field, data[i] * conversion)

        # Compute the characteristic entropy using eq 3 in the paper
        self.K500 = unyt_quantity(106, 'keV*cm**2') * \
                    data[5] ** (2 / 3) * \
                    self.fb0 ** (-2 / 3) * \
                    self.ez_function(self.z) ** (-2 / 3) * \
                    self.hconv ** (-4 / 3)

    def process_entropy_profiles(self):
        field_names = (
            'Cluster_name '
            'K0 Delta_K0 '
            'KR100 Delta_KR100 '
            'alpha Delta_alpha').split()
        data = []

        with open(f'{repository_dir}/pratt2010_profiles.dat') as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith('#') and not line.isspace():
                    line_data = line.split()
                    for i, element_data in enumerate(line_data):
                        if element_data.strip() == 'none':
                            # If no data, replace with Nan
                            line_data[i] = np.nan
                        elif re.search('[a-df-zA-Z]', element_data):
                            # If contains letters, remove white spaces
                            line_data[i] = element_data.strip()
                        else:
                            line_data[i] = float(element_data.strip())
                    data.append(line_data)

        data = list(map(list, itertools.zip_longest(*data, fillvalue=None)))
        conversion_factors = [
            None,
            keV * cm ** 2,
            keV * cm ** 2,
            keV * cm ** 2,
            keV * cm ** 2,
            1,
            1,
        ]

        for i, field in enumerate(data):
            if isinstance(data[i][0], str):
                data[i] = np.array(field)
            else:
                data[i] = np.array(field) * conversion_factors[i]

        radial_range = np.logspace(-2, 0, 1001)
        profiles = np.zeros((len(data[0]), len(radial_range)))
        for i, (K0, K100, alpha, redshift, R500) in enumerate(zip(
                data[1],
                data[3],
                data[5],
                self.z,
                self.R500
        )):
            if np.isnan(alpha):
                profiles[i] = np.nan
            else:
                radius = np.logspace(np.log10(0.01 * R500.v), np.log10(R500.v), len(radial_range))
                profiles[i] = K0 + K100 * (radius / 100 / self.hconv) ** alpha

        self.entropy_profiles = unyt_array(profiles, keV * cm ** 2)
        self.radial_bins = unyt_array(radial_range, dimensionless)
        self.entropy_profiles_k500_rescaled = False

    def combine_entropy_profiles(
            self,
            m500_limits: Tuple[unyt_quantity] = (
                    1e10 * Solar_Mass,
                    1e17 * Solar_Mass
            ),
            k500_rescale: bool = True
    ):

        if k500_rescale:
            self.entropy_profiles /= self.K500[:, None]
            self.entropy_profiles_k500_rescaled = True

        mass_bin = np.where(
            (self.M500.value > m500_limits[0].value) &
            (self.M500.value < m500_limits[1].value)
        )[0]

        bin_median = np.nanmedian(self.entropy_profiles[mass_bin], axis=0)
        bin_perc16 = np.nanpercentile(self.entropy_profiles[mass_bin], 16, axis=0)
        bin_perc84 = np.nanpercentile(self.entropy_profiles[mass_bin], 84, axis=0)

        return bin_median, bin_perc16, bin_perc84

    def quick_display(self, **kwargs):

        bin_median, bin_perc16, bin_perc84 = self.combine_entropy_profiles(**kwargs)

        # Display the catalogue data
        for profile in self.entropy_profiles:
            plt.plot(self.radial_bins, profile, c='k', alpha=0.1)

        plt.fill_between(
            self.radial_bins,
            bin_perc16,
            bin_perc84,
            color='aqua', alpha=0.85, linewidth=0
        )
        plt.plot(self.radial_bins, bin_median, c='k')

        plt.xlabel(r'$r/r_{500}$')
        y_label = r'$E(z)^{4/3} K$ [keV cm$^2$]'
        if self.entropy_profiles_k500_rescaled:
            y_label = r'$K / K_{500}$'
        plt.ylabel(y_label)
        plt.title(f"REXCESS sample - {self.citation}")
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        plt.close()
