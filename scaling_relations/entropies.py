import os.path
import numpy as np
from warnings import warn
from unyt import kb, mp, Mpc
from scipy.interpolate import interp1d

from .halo_property import HaloProperty, histogram_unyt
from .spherical_overdensities import SphericalOverdensities
from register import Zoom, Tcut_halogas, default_output_directory, args

# Constants
mean_molecular_weight = 0.59
mean_atomic_weight_per_free_electron = 1.14


class Entropies(HaloProperty):

    def __init__(self):
        super().__init__()

        self.labels = ['k30kpc', 'k2500', 'k1500', 'k1000', 'k500', 'k200']

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'entropies_{args.redshift_index:04d}.pkl'
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
        r2500 = vr_data.spherical_overdensities.r_2500_rhocrit[0].to('Mpc')

        kwarg_parser = dict(zoom_obj=zoom_obj, path_to_snap=path_to_snap, path_to_catalogue=path_to_catalogue)

        try:
            r1000 = vr_data.spherical_overdensities.r_1000_rhocrit[0].to('Mpc')
        except:
            r1000 = SphericalOverdensities(density_contrast=1000).process_single_halo(**kwarg_parser)[0]

        try:
            r1500 = vr_data.spherical_overdensities.r_1500_rhocrit[0].to('Mpc')
        except:
            r1500 = SphericalOverdensities(density_contrast=1500).process_single_halo(**kwarg_parser)[0]

        r200 = vr_data.radii.r_200crit[0].to('Mpc')

        sw_data.gas.radial_distances.convert_to_physical()

        # Select hot gas within sphere
        mask = np.where(
            (sw_data.gas.radial_distances <= 2 * r500) &
            (sw_data.gas.temperatures > Tcut_halogas) &
            (sw_data.gas.fofgroup_ids == 1)
        )[0]

        sw_data.gas.radial_distances = sw_data.gas.radial_distances[mask]
        sw_data.gas.masses = sw_data.gas.masses[mask]
        sw_data.gas.temperatures = sw_data.gas.temperatures[mask]

        radial_distances_scaled = sw_data.gas.radial_distances / r500

        # Define radial bins and shell volumes
        lbins = np.logspace(
            np.log10(radial_distances_scaled.min().value),
            np.log10(radial_distances_scaled.max().value),
            300
        ) * radial_distances_scaled.units
        radial_bin_centres = 10.0 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * radial_distances_scaled.units
        volume_shell = (4. * np.pi / 3.) * (r500 ** 3) * ((lbins[1:]) ** 3 - (lbins[:-1]) ** 3)

        mass_weights, _ = histogram_unyt(radial_distances_scaled, bins=lbins, weights=sw_data.gas.masses)
        mass_weights[mass_weights == 0] = np.nan  # Replace zeros with Nans
        density_profile = mass_weights / volume_shell
        number_density_profile = (density_profile.to('g/cm**3') / (mp * mean_molecular_weight)).to('cm**-3')

        mass_weighted_temperatures = (sw_data.gas.temperatures * kb).to('keV') * sw_data.gas.masses
        temperature_weights, _ = histogram_unyt(radial_distances_scaled, bins=lbins, weights=mass_weighted_temperatures)
        temperature_weights[temperature_weights == 0] = np.nan  # Replace zeros with Nans
        temperature_profile = temperature_weights / mass_weights  # kBT in units of [keV]

        entropy_profile = temperature_profile / number_density_profile ** (2 / 3)

        print(entropy_profile)

        entropy_interpolate = interp1d(radial_bin_centres * r500, entropy_profile,
                                       kind='quadratic', fill_value='extrapolate')

        k30kpc = entropy_interpolate(0.03 * Mpc) * entropy_profile.units
        k500 = entropy_interpolate(r500) * entropy_profile.units
        k2500 = entropy_interpolate(r2500) * entropy_profile.units
        k1500 = entropy_interpolate(r1500) * entropy_profile.units
        k1000 = entropy_interpolate(r1000) * entropy_profile.units
        k200 = entropy_interpolate(r200) * entropy_profile.units

        return k30kpc, k2500, k1500, k1000, k500, k200

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):

        return self._read_catalogue(self.filename)
