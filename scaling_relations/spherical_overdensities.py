import os.path
import numpy as np
from warnings import warn
from scipy.interpolate import interp1d
from unyt import (
    kb, mp, Mpc, Solar_Mass,
    unyt_quantity, unyt_array
)

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
            f'spherical_overdensities_dc{density_contrast:.0f}_{args.redshift_index:04d}.pkl'
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

        if args.debug:
            print((
                f"[{self.__class__.__name__}] Density contrast: {self.density_contrast:.2f}\n"
                f"[{self.__class__.__name__}] Snap: {os.path.basename(path_to_snap)}\n"
                f"[{self.__class__.__name__}] Catalog: {os.path.basename(path_to_catalogue)}"
            ))

        sw_data, vr_data = self.get_handles_from_zoom(zoom_obj, path_to_snap, path_to_catalogue, **kwargs)

        # Try to import r500 from the catalogue.
        # If not there (and needs to be computed), assume 1 Mpc for the spatial mask.
        try:
            aperture_search = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc') * 3
        except AttributeError as err:
            aperture_search = unyt_quantity(1, Mpc) * 3
            if args.debug:
                print(err, f"[{self.__class__.__name__}] Setting aperture_search = 3. Mpc.", sep='\n')

        sw_data.gas.radial_distances.convert_to_physical()
        sw_data.dark_matter.radial_distances.convert_to_physical()

        # Calculate the critical density
        rho_crit = unyt_quantity(
            sw_data.metadata.cosmology.critical_density(sw_data.metadata.z).value, 'g/cm**3'
        ).to('Msun/Mpc**3')

        radial_distances_collect = [
            sw_data.gas.radial_distances,
            sw_data.dark_matter.radial_distances,
        ]
        if sw_data.metadata.n_stars > 0:
            sw_data.stars.radial_distances.convert_to_physical()
            radial_distances_collect.append(sw_data.stars.radial_distances)
        elif args.debug:
            print(f"[{self.__class__.__name__}] stars not detected.")

        if sw_data.metadata.n_black_holes > 0:
            sw_data.black_holes.radial_distances.convert_to_physical()
            radial_distances_collect.append(sw_data.black_holes.radial_distances)
        elif args.debug:
            print(f"[{self.__class__.__name__}] black_holes not detected.")

        radial_distances = np.r_[[*radial_distances_collect]][0]

        sw_data.gas.masses.convert_to_physical()
        sw_data.dark_matter.masses.convert_to_physical()
        masses_collect = [
            sw_data.gas.masses,
            sw_data.dark_matter.masses,
        ]
        if sw_data.metadata.n_stars > 0:
            sw_data.stars.masses.convert_to_physical()
            masses_collect.append(sw_data.stars.masses)
        elif args.debug:
            print(f"[{self.__class__.__name__}] stars not detected.")

        if sw_data.metadata.n_black_holes > 0:
            sw_data.black_holes.subgrid_masses.convert_to_physical()
            masses_collect.append(sw_data.black_holes.subgrid_masses)
        elif args.debug:
            print(f"[{self.__class__.__name__}] black_holes not detected.")

        masses = np.r_[[*masses_collect]][0]

        try:
            fof_ids_collect = [
                sw_data.gas.fofgroup_ids,
                sw_data.dark_matter.fofgroup_ids,
            ]
            if sw_data.metadata.n_stars > 0:
                fof_ids_collect.append(sw_data.stars.fofgroup_ids)
            elif args.debug:
                print(f"[{self.__class__.__name__}] stars not detected.")

            if sw_data.metadata.n_black_holes > 0:
                fof_ids_collect.append(sw_data.black_holes.fofgroup_ids)
            elif args.debug:
                print(f"[{self.__class__.__name__}] black_holes not detected.")

            fof_ids = np.r_[[*fof_ids_collect]][0]

            # Select all particles within sphere
            mask = np.where(
                (radial_distances <= aperture_search) &
                (fof_ids == 1)
            )[0]

            del fof_ids

        except AttributeError as err:
            print(
                err,
                f"[{self.__class__.__name__}] Select particles only by radial distance.",
                sep='\n'
            )
            mask = np.where(radial_distances <= aperture_search)[0]

        radial_distances = unyt_array(radial_distances.value, radial_distances.units)[mask]
        radial_distances /= aperture_search
        masses = unyt_array(masses.value, masses.units)[mask]

        del mask

        # Define radial bins and shell volumes
        lbins = np.logspace(
            np.log10(radial_distances.min()) - 1e-6,
            np.log10(radial_distances.max()) + 1e-6,
            500
        ) * radial_distances.units
        radial_bin_centres = 10.0 ** (0.5 * np.log10(lbins[1:] * lbins[:-1])) * radial_distances.units
        volume_sphere = (4. * np.pi / 3.) * lbins[1:] ** 3 * aperture_search ** 3

        print(lbins)
        print(radial_bin_centres)
        print(volume_sphere)

        mass_weights, _ = histogram_unyt(radial_distances, bins=lbins, weights=masses)
        mass_weights[mass_weights == 0] = np.nan  # Replace zeros with Nans
        cumulative_mass_profile = cumsum_unyt(mass_weights)
        density_profile = cumulative_mass_profile / volume_sphere / rho_crit
        print(density_profile)
        # For better stability, clip the initial 5% of the profile
        clip = int((len(lbins) - 1) / 20)

        density_interpolate = interp1d(
            np.log10(density_profile[clip:].value),
            np.log10(radial_bin_centres[clip:].value),
            kind='linear'
        )
        mass_interpolate = interp1d(
            np.log10(radial_bin_centres[clip:].value),
            np.log10(cumulative_mass_profile[clip:].value),
            kind='linear'
        )

        r_delta = 10 ** density_interpolate(np.log10(self.density_contrast)) * Mpc
        m_delta = 10 ** mass_interpolate(np.log10(r_delta)) * mass_weights.units

        if args.debug:
            print((
                f"[{self.__class__.__name__}] r_delta: {r_delta:.2f}\n"
                f"[{self.__class__.__name__}] m_delta: {m_delta.to(Solar_Mass):.2E}"
            ))
            assert r_delta > 0
            assert m_delta > 0

        return r_delta, m_delta

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):

        return self._read_catalogue(self.filename)


class SODelta2500(SphericalOverdensities):
    def __init__(self,
                 zoom_obj: Zoom = None,
                 path_to_snap: str = None,
                 path_to_catalogue: str = None,
                 **kwargs):
        super().__init__(density_contrast=2500.)

        r_delta, m_delta = self.process_single_halo(
            zoom_obj=zoom_obj,
            path_to_snap=path_to_snap,
            path_to_catalogue=path_to_catalogue,
            **kwargs
        )
        self.r2500 = r_delta
        self.m2500 = m_delta

    def get_r2500(self):
        return self.r2500

    def get_m2500(self):
        return self.m2500


class SODelta500(SphericalOverdensities):
    def __init__(self,
                 zoom_obj: Zoom = None,
                 path_to_snap: str = None,
                 path_to_catalogue: str = None,
                 **kwargs):
        super().__init__(density_contrast=500.)

        r_delta, m_delta = self.process_single_halo(
            zoom_obj=zoom_obj,
            path_to_snap=path_to_snap,
            path_to_catalogue=path_to_catalogue,
            **kwargs
        )
        self.r500 = r_delta
        self.m500 = m_delta

    def get_r500(self):
        return self.r500

    def get_m500(self):
        return self.m500


class SODelta200(SphericalOverdensities):
    def __init__(self,
                 zoom_obj: Zoom = None,
                 path_to_snap: str = None,
                 path_to_catalogue: str = None,
                 **kwargs):
        super().__init__(density_contrast=200.)

        r_delta, m_delta = self.process_single_halo(
            zoom_obj=zoom_obj,
            path_to_snap=path_to_snap,
            path_to_catalogue=path_to_catalogue,
            **kwargs
        )
        self.r200 = r_delta
        self.m200 = m_delta

    def get_r200(self):
        return self.r200

    def get_m200(self):
        return self.m200
