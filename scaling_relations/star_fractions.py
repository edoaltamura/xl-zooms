import os.path
import numpy as np
from warnings import warn

from .halo_property import HaloProperty
from register import Zoom, default_output_directory, xlargs


class StarFractions(HaloProperty):

    def __init__(self):
        super().__init__()

        self.labels = ['m_star', 'f_star']

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'star_fractions_{xlargs.mass_estimator:s}_{xlargs.redshift_index:04d}.pkl'
        )

    def check_value(self, value):

        if value >= 1:
            raise RuntimeError((
                f"The value for {self.labels[1]} must be between 0 and 1. "
                f"Got {value} instead."
            ))
        elif 0.5 < value < 1:
            warn(f"The value for {self.labels[1]} seems too high: {value}", RuntimeWarning)

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

        sw_data.stars.radial_distances.convert_to_physical()
        sw_data.stars.masses.convert_to_physical()

        # Select hot gas within sphere
        mask = np.where(
            (sw_data.stars.radial_distances <= r500) &
            (sw_data.stars.fofgroup_ids == 1)
        )[0]
        m_star500 = np.sum(sw_data.stars.masses[mask])
        m_star500 = m_star500.to('Msun')
        star_fraction = m_star500 / m500

        self.check_value(star_fraction)
        return m_star500, star_fraction

    def compute_from_vr(
            self,
            zoom_obj: Zoom = None,
            zoom_name: str = None
    ):
        catalogue_file = os.path.join(
            default_output_directory,
            'intermediate',
            f'vrproperties_{xlargs.redshift_index:04d}.pkl'
        )
        data = self._get_zoom_from_catalogue(catalogue_file, zoom_obj=zoom_obj, zoom_name=zoom_name)

        m500 = data['m500']
        m_star500c = data['m_star500c']
        star_fraction = m_star500c / m500

        self.check_value(star_fraction)
        return m_star500c, star_fraction

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):

        return self._read_catalogue(self.filename)

    def get_zoom_from_catalogue(self, **kwargs):

        return self._get_zoom_from_catalogue(self.filename, **kwargs)
