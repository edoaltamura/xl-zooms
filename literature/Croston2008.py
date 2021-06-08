import sys
import os
from unyt import (
    unyt_quantity, unyt_array,
    keV, cm, dimensionless, Solar_Mass, erg, s, kpc, mp
)
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from .cosmology import Article, repository_dir

mean_molecular_weight = 0.5954
mean_atomic_weight_per_free_electron = 1.14

comment = (
    "REXCESS sample."
)


class Croston2008(Article):

    def __init__(self, **cosmo_kwargs):
        super().__init__(
            citation="Croston et al. (2008)",
            comment=comment,
            bibcode="2008A&A...487..431C",
            hyperlink='https://ui.adsabs.harvard.edu/abs/2008A%26A...487..431C/abstract',
            **cosmo_kwargs
        )

        self.hconv = 0.70 / self.h
        self.cluster_data = []
        self.process_profiles()

    @staticmethod
    def log_cluster_data(file_path: str) -> dict:
        field_names = 'r_kpc r_r500 n_e n_e_sigma'.split()
        field_units = 'kpc dimensionless cm**-3 cm**-3'.split()
        cluster_data = {}
        cluster_data['name'] = os.path.basename(file_path)[:-4]

        numerical_data = np.loadtxt(file_path).T
        for field_name, field_values, units in zip(field_names, numerical_data, field_units):
            cluster_data[field_name] = unyt_array(field_values, units)

        return cluster_data

    def process_profiles(self):
        for file_path in os.listdir(
            os.path.join(repository_dir, 'Croston2008_profiles')
        ):
            if file_path.startswith('RXC'):
                cluster = self.log_cluster_data(
                    os.path.join(repository_dir, 'Croston2008_profiles', file_path)
                )
                self.cluster_data.append(cluster)

    def compute_gas_mass(self):
        for cluster in self.cluster_data:
            radial_bin_centres = 10.0 ** (0.5 * np.log10(cluster['r_kpc'][1:] * cluster['r_kpc'][:-1])) * kpc

            radius_interpolate = interp1d(
                cluster['r_kpc'].value,
                cluster['n_e'].value,
                kind='linear',
                fill_value='extrapolate'
            )
            n_e_bin_centres = radius_interpolate(radial_bin_centres) * cluster['n_e'].units

            gas_density_bin_centres = n_e_bin_centres * (mp * mean_molecular_weight) / mean_atomic_weight_per_free_electron
            gas_density_bin_centres.convert_to_units('Msun/kpc**3')
            volume_shells = 4 * np.pi / 3 * (cluster['r_kpc'][1:] ** 3 - cluster['r_kpc'][:-1] ** 3)
            print(gas_density_bin_centres)
            gas_mass_bin_centres = gas_density_bin_centres * volume_shells

            radius_interpolate = interp1d(
                n_e_bin_centres.value,
                gas_mass_bin_centres.value,
                kind='linear',
                fill_value='extrapolate'
            )
            gas_mass_bin_edges = radius_interpolate(cluster['r_kpc'])

            gas_mass_bin_edges = np.cumsum(gas_mass_bin_edges) * gas_mass_bin_centres.units
            gas_mass_bin_edges.astype(np.float64).convert_to_units('Msun')
            cluster['Mgas'] = gas_mass_bin_edges

    def interpolate_r_r500(self, r_r500_new: np.ndarray):

        for cluster in self.cluster_data:

            for field in ['r_kpc', 'n_e', 'n_e_sigma']:
                radius_interpolate = interp1d(
                    cluster['r_r500'],
                    cluster[field],
                    kind='linear',
                    fill_value='extrapolate'
                )
                cluster[field] = radius_interpolate(r_r500_new) * cluster[field].units

            cluster['r_r500'] = r_r500_new * cluster['r_r500'].units

    # def combine_entropy_profiles(
    #         self,
    #         m500_limits: Tuple[unyt_quantity] = (
    #                 1e10 * Solar_Mass,
    #                 1e17 * Solar_Mass
    #         ),
    #         k500_rescale: bool = True
    # ):
    #
    #     if k500_rescale:
    #         self.entropy_profiles /= self.K500[:, None]
    #         self.entropy_profiles_k500_rescaled = True
    #
    #     mass_bin = np.where(
    #         (self.M500.value > m500_limits[0].value) &
    #         (self.M500.value < m500_limits[1].value)
    #     )[0]
    #
    #     bin_median = np.nanmedian(self.entropy_profiles[mass_bin], axis=0)
    #     bin_perc16 = np.nanpercentile(self.entropy_profiles[mass_bin], 16, axis=0)
    #     bin_perc84 = np.nanpercentile(self.entropy_profiles[mass_bin], 84, axis=0)
    #
    #     return bin_median, bin_perc16, bin_perc84

    def quick_display(self, **kwargs):

        # bin_median, bin_perc16, bin_perc84 = self.combine_entropy_profiles(**kwargs)
        self.compute_gas_mass()
        # Display the catalogue data
        for cluster in self.cluster_data:
            plt.plot(cluster['r_r500'], cluster['Mgas'], c='k', alpha=0.1)

        # plt.fill_between(
        #     self.radial_bins,
        #     bin_perc16,
        #     bin_perc84,
        #     color='aqua', alpha=0.85, linewidth=0
        # )
        # plt.plot(self.radial_bins, bin_median, c='k')

        plt.xlabel(r'$r/r_{500}$')
        y_label = r'$E(z)^{4/3} K$ [keV cm$^2$]'
        plt.ylabel(y_label)
        plt.title(f"REXCESS sample - {self.citation}")
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        plt.close()
