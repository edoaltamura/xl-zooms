import os.path

from .halo_property import HaloProperty
from register import Zoom, default_output_directory, args


class VRProperties(HaloProperty):

    def __init__(self):
        super().__init__()

        self.labels = ['r2500', 'r1000', 'r500', 'r200', 'm2500', 'm1000', 'm500', 'm200', 'm_star500c', 'm_star30kpc']

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'vrproperties_{args.redshift_index:04d}.pkl'
        )

    def process_single_halo(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None,
            **kwargs
    ):
        _, vr_data = self.get_handles_from_zoom(zoom_obj, path_to_snap, path_to_catalogue, **kwargs)

        m500 = vr_data.spherical_overdensities.mass_500_rhocrit[0].to('Msun')
        r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc')
        m2500 = vr_data.spherical_overdensities.mass_2500_rhocrit[0].to('Msun')
        r2500 = vr_data.spherical_overdensities.r_2500_rhocrit[0].to('Mpc')
        m1000 = vr_data.spherical_overdensities.mass_1000_rhocrit[0].to('Msun')
        r1000 = vr_data.spherical_overdensities.r_1000_rhocrit[0].to('Mpc')
        m200 = vr_data.masses.m_200crit[0].to('Msun')
        r200 = vr_data.radii.r_200crit[0].to('Mpc')

        m_star500c = vr_data.spherical_overdensities.mass_star_500_rhocrit[0].to('Msun')
        m_star30kpc = vr_data.apertures.mass_star_50_kpc[0].to('Msun')

        return r2500, r1000, r500, r200, m2500, m1000, m500, m200, m_star500c, m_star30kpc

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):

        return self._read_catalogue(self.filename)

    def get_zoom_from_catalogue(self, **kwargs):

        return self._get_zoom_from_catalogue(self.filename, **kwargs)


