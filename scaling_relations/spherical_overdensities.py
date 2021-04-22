import os.path
import numpy as np
from warnings import warn
from scipy.interpolate import interp1d
from unyt import kb, mp, Mpc, Solar_Mass, unyt_quantity

from .halo_property import HaloProperty, histogram_unyt, cumsum_unyt
from register import Zoom, default_output_directory, args


class SphericalOverdensities(HaloProperty):

    def __init__(self, density_contrast: float):
        super().__init__()

        self.density_contrast = density_contrast
        self.labels = [f'r_{density_contrast:.0f}', f'm_{density_contrast:.0f}']

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'spherical_overdensities_{args.redshift_index:04d}.pkl'
        )

    def check_value(self, value):

        if value >= 1:
            raise RuntimeError((
                f"The value for {self.labels[0]} must be between 0 and 1. "
                f"Got {value} instead."
            ))
        elif 0.5 < value < 1:
            warn(f"The value for {self.labels[0]} seems too high: {value}", RuntimeWarning)

    def process_single_halo(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None,
            **kwargs
    ):
        sw_data, vr_data = self.get_handles_from_zoom(zoom_obj, path_to_snap, path_to_catalogue, **kwargs)

        r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc')

        sw_data.gas.radial_distances.convert_to_physical()
        # sw_data.gas.masses.convert_to_physical()
        sw_data.dark_matter.radial_distances.convert_to_physical()
        # sw_data.dark_matter.masses.convert_to_physical()
        sw_data.stars.radial_distances.convert_to_physical()
        # sw_data.stars.masses.convert_to_physical()
        sw_data.black_holes.radial_distances.convert_to_physical()
        # sw_data.black_holes.subgrid_masses.convert_to_physical()

        # Calculate the critical density for the cross-hair marker
        rho_crit = unyt_quantity(
            sw_data.metadata.cosmology.critical_density(sw_data.metadata.z).value, 'g/cm**3'
        ).to('Msun/Mpc**3')

        radial_distances = np.r_[
            sw_data.gas.radial_distances,
            sw_data.dark_matter.radial_distances,
            sw_data.stars.radial_distances,
            sw_data.black_holes.radial_distances
        ]

        masses = np.r_[
            sw_data.gas.masses,
            sw_data.dark_matter.masses,
            sw_data.stars.masses,
            sw_data.black_holes.subgrid_masses
        ]

        fof_ids = np.r_[
            sw_data.gas.fofgroup_ids,
            sw_data.dark_matter.fofgroup_ids,
            sw_data.stars.fofgroup_ids,
            sw_data.black_holes.fofgroup_ids
        ]

        # Select all particles within sphere
        mask = np.where(
            (radial_distances <= 3 * r500) &
            (fof_ids == 1)
        )[0]

        del fof_ids

        radial_distances = radial_distances[mask] * Mpc / r500
        masses = masses[mask] * 1e10 * Solar_Mass

        # Define radial bins and shell volumes
        lbins = np.logspace(
            np.log10(radial_distances.min().value / 1.1),
            np.log10(radial_distances.max().value * 1.1),
            50
        ) * radial_distances.units
        radial_bin_centres = 10.0 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * radial_distances.units
        volume_sphere = (4. * np.pi / 3.) * r500 ** 3 * lbins[1:] ** 3

        mass_weights, _ = histogram_unyt(radial_distances, bins=lbins, weights=masses)
        cumulative_mass_profile = cumsum_unyt(mass_weights)
        density_profile = cumulative_mass_profile / volume_sphere / rho_crit

        print(density_profile)

        density_interpolate = interp1d(density_profile, radial_bin_centres * r500,
                                       kind='quadratic', fill_value='extrapolate')

        mass_interpolate = interp1d(radial_bin_centres * r500, cumulative_mass_profile,
                                    kind='quadratic', fill_value='extrapolate')

        r_delta = density_interpolate(self.density_contrast) * r500.units
        m_delta = mass_interpolate(r_delta) * mass_weights.units
        print(r500, vr_data.spherical_overdensities.mass_500_rhocrit[0].to('Msun'))
        return r_delta, m_delta

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):

        return self._read_catalogue(self.filename)
