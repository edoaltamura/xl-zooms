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
from .Pratt2010 import Pratt2010

mean_molecular_weight = 0.5954
mean_atomic_weight_per_free_electron = 1.14

comment = (
    "REXCESS sample."
)


class Croston2008(Article):

    def __init__(self, disable_cosmo_conversion: bool = False, **cosmo_kwargs):
        super().__init__(
            citation="Croston et al. (2008)",
            comment=comment,
            bibcode="2008A&A...487..431C",
            hyperlink='https://ui.adsabs.harvard.edu/abs/2008A%26A...487..431C/abstract',
            **cosmo_kwargs
        )

        self.hconv = 0.70 / self.h
        if disable_cosmo_conversion:
            self.hconv = 1.

        self.pratt2010 = Pratt2010(disable_cosmo_conversion=disable_cosmo_conversion)

        self.cluster_data = []
        self.process_profiles()
        self.compute_gas_mass()
        self.estimate_total_mass()
        self.compute_gas_fraction()

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
        for file_path in self.pratt2010.Cluster_name:
            cluster = self.log_cluster_data(
                os.path.join(repository_dir, 'Croston2008_profiles', f"{file_path}.txt")
            )
            self.cluster_data.append(cluster)

    def compute_gas_mass(self):
        for cluster in self.cluster_data:
            # radial_bin_centres = 10.0 ** (0.5 * np.log10(cluster['r_kpc'][1:] * cluster['r_kpc'][:-1])) * kpc
            radial_bin_centres = (cluster['r_kpc'][1:] + cluster['r_kpc'][:-1]) / 2

            # Find value of the electron number density at bin centres by interpolating
            radius_interpolate = interp1d(
                cluster['r_kpc'].value,
                cluster['n_e'].value,
                kind='linear',
                fill_value='extrapolate'
            )
            n_e_bin_centres = radius_interpolate(radial_bin_centres) * cluster['n_e'].units

            # Convert electron number density to gas density
            gas_density_bin_centres = n_e_bin_centres * mp / mean_atomic_weight_per_free_electron
            gas_density_bin_centres.astype(np.float64).convert_to_units('Msun/kpc**3')
            volume_shells = 4 * np.pi / 3 * (cluster['r_kpc'][1:] ** 3 - cluster['r_kpc'][:-1] ** 3)
            gas_mass_bin_centres = gas_density_bin_centres * volume_shells

            # Interpolate/extrapolate results back to the bin edges
            radius_interpolate = interp1d(
                radial_bin_centres.value,
                gas_mass_bin_centres.value,
                kind='linear',
                fill_value='extrapolate'
            )
            gas_mass_bin_edges = radius_interpolate(cluster['r_kpc'])

            # Cumulate mass in shells to find the gas mass enclosed
            gas_mass_bin_edges = np.cumsum(gas_mass_bin_edges) * gas_mass_bin_centres.units
            gas_mass_bin_edges.astype(np.float64).convert_to_units('Msun')
            cluster['Mgas'] = gas_mass_bin_edges.to('Msun')

    @staticmethod
    def nfw_factor(scale_radius, r):
        return np.log((scale_radius + r) / scale_radius) - r / (r + scale_radius)

    def estimate_total_mass(self):

        average_concentration = 3.2

        compton_like_yx = self.pratt2010.YX_R500

        M500 = 10 ** 14.567 * Solar_Mass / 0.7 * self.hconv * \
               (compton_like_yx / unyt_quantity(2e14, 'Msun*keV')) ** 0.561 \
               * self.ez_function(self.pratt2010.z) ** (-2 / 5)

        for cluster, m500, r500 in zip(self.cluster_data, M500, self.pratt2010.R500):
            scale_radius = r500 / average_concentration
            r = cluster['r_kpc']

            rho_0 = m500 / (4 * np.pi * scale_radius ** 3 * self.nfw_factor(scale_radius, r500))
            nfw_enclosed_mass = 4 * np.pi * rho_0 * scale_radius ** 3 * self.nfw_factor(scale_radius, r)
            cluster['Mdm_nfw'] = nfw_enclosed_mass
            cluster['m500'] = m500

    def compute_gas_fraction(self):
        for cluster in self.cluster_data:
            assert 'Mgas' in cluster
            assert 'Mdm_nfw' in cluster
            cluster['f_g'] = cluster['Mgas'] / cluster['Mdm_nfw']

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

    def quick_display(self, quantity: str = 'f_g'):
        self.compute_gas_mass()
        self.estimate_total_mass()
        self.compute_gas_fraction()

        # Display the catalogue data
        for cluster in self.cluster_data:

            for value in cluster['Mgas']:
                if value < 1.e7:
                    print(cluster['name'])

            plt.plot(cluster['r_r500'], cluster[quantity], c='k',
                     alpha=0.7, lw=0.3)

        # plt.axhline(self.fb0)
        plt.xlabel(r'$r/r_{500}$')
        y_label = r'$h_{70}^{-3/2}\ f_g (<R)$'
        plt.ylabel(y_label)
        plt.title(f"REXCESS sample - {self.citation}")
        plt.xscale('log')
        plt.yscale('log')
        # plt.ylim(0, self.fb0 * 1.1)
        plt.show()
        plt.close()
