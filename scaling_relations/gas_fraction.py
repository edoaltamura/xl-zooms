import unyt
import numpy as np

from .halo_property import HaloProperty
from register import Zoom, Tcut_halogas


class GasFraction(HaloProperty):

    def __init__(self):
        super().__init__()

    def process_single_halo(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None,
            **kwargs
    ):
        sw_data, vr_data = self.get_handles_from_zoom(zoom_obj, path_to_snap, path_to_catalogue, **kwargs)

        m500 = vr_data.spherical_overdensities.mass_500_rhocrit[0].to('Msun')
        r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc')

        # Select hot gas within sphere
        mask = np.where(
            (sw_data.gas.radial_distances <= r500) &
            (sw_data.gas.temperatures > Tcut_halogas) &
            (sw_data.gas.fofgroup_ids == 1)
        )[0]
        mhot500 = np.sum(sw_data.gas.masses[mask])
        mhot500 = mhot500.to(unyt.Solar_Mass)
        gas_fraction = mhot500 / m500

        return gas_fraction