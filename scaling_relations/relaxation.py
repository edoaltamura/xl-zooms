import os.path
import numpy as np
from unyt import kb, hydrogen_mass

from .halo_property import HaloProperty
from register import Zoom, Tcut_halogas, default_output_directory, xlargs


class Relaxation(HaloProperty):

    def __init__(self):
        super().__init__()

        self.labels = ['Ekin', 'Etherm']

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'relaxation_{xlargs.mass_estimator:s}_{xlargs.redshift_index:04d}.pkl'
        )

    @staticmethod
    def check_value(value):
        if value >= 5 or value <= 0:
            raise RuntimeError((
                f"The value for Ekin / Etherm should be between 0 and 1. "
                f"Got {value}."
            ))

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
        sw_data.gas.temperatures.convert_to_physical()
        sw_data.gas.velocities.convert_to_physical()
        sw_data.gas.masses.convert_to_physical()

        # Select hot gas within sphere
        mask = np.where(
            (sw_data.gas.radial_distances <= r500) &
            (sw_data.gas.temperatures > Tcut_halogas) &
            (sw_data.gas.fofgroup_ids == 1)
        )[0]

        # Compute kinetic energy in the halo's rest frame
        gas_velocities = sw_data.gas.velocities
        linear_momentum = gas_velocities[mask] * sw_data.gas.masses[mask, None]
        gas_mass = np.sum(sw_data.gas.masses[mask])
        peculiar_velocity = np.sum(linear_momentum, axis=0) / gas_mass
        gas_velocities[:, 0] -= peculiar_velocity[0]
        gas_velocities[:, 1] -= peculiar_velocity[1]
        gas_velocities[:, 2] -= peculiar_velocity[2]

        # Compute square of the modulus (einsum faster than linalg.norm)
        v2 = np.einsum('...i,...i', gas_velocities, gas_velocities) * (gas_velocities.units ** 2)
        kbT = kb * sw_data.gas.temperatures * sw_data.gas.masses

        Ekin = np.sum(0.5 * sw_data.gas.masses[mask] * v2[mask]).to("1.e10*Mpc**2*Msun/Gyr**2")
        Etherm = np.sum(1.5 * kbT[mask] / (hydrogen_mass / 1.16)).to("1.e10*Mpc**2*Msun/Gyr**2")

        self.check_value(Ekin / Etherm)
        return Ekin, Etherm

    def process_catalogue(self):
        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):
        return self._read_catalogue(self.filename)

    def get_zoom_from_catalogue(self, **kwargs):
        return self._get_zoom_from_catalogue(self.filename, **kwargs)
