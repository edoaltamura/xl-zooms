import os.path

from .halo_property import HaloProperty
from register import Zoom, default_output_directory, xlargs


class VRProperties(HaloProperty):

    def __init__(self):
        super().__init__()

        self.labels = [
            'r2500', 'r1000', 'r500', 'r200', 'm2500', 'm1000', 'm500', 'm200', 'm_star500c', 'm_star30kpc',
            'm_star100kpc', 'sfr_100kpc', 'm_bh100kpc'
        ]

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'vrproperties_{xlargs.redshift_index:04d}.pkl'
        )

    def process_single_halo(
            self,
            zoom_obj: Zoom = None,
            path_to_catalogue: str = None,
    ):
        vr_data = self.get_vr_handle(zoom_obj, path_to_catalogue)

        try:
            m500 = vr_data.spherical_overdensities.mass_500_rhocrit[0].to('Msun')
            r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc')
            m2500 = vr_data.spherical_overdensities.mass_2500_rhocrit[0].to('Msun')
            r2500 = vr_data.spherical_overdensities.r_2500_rhocrit[0].to('Mpc')
            m1000 = vr_data.spherical_overdensities.mass_1000_rhocrit[0].to('Msun')
            r1000 = vr_data.spherical_overdensities.r_1000_rhocrit[0].to('Mpc')
            m200 = vr_data.masses.mass_200crit[0].to('Msun')
            r200 = vr_data.radii.r_200crit[0].to('Mpc')

            m_star500c = vr_data.spherical_overdensities.mass_star_500_rhocrit[0].to('Msun')
            m_star30kpc = vr_data.apertures.mass_star_50_kpc[0].to('Msun')
            m_star100kpc = vr_data.apertures.mass_star_100_kpc[0].to('Msun')
            sfr_100kpc = vr_data.apertures.sfr_gas_100_kpc[0].to('Msun/Gyr')
            m_bh100kpc = vr_data.apertures.mass_bh_100_kpc[0].to('Msun')

        except AttributeError:
            m500 = None
            r500 = None
            m2500 = None
            r2500 = None
            m1000 = None
            r1000 = None
            m200 = None
            r200 = None

            m_star500c = None
            m_star30kpc = None
            m_star100kpc = None
            sfr_100kpc = None
            m_bh100kpc = None

        return (
            r2500, r1000, r500, r200, m2500, m1000, m500, m200, m_star500c, m_star30kpc, m_star100kpc,
            sfr_100kpc, m_bh100kpc
        )

    def process_catalogue(self, save_to_file: bool = False):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)

        if save_to_file:
            self.dump_to_pickle(self.filename, catalogue)

        return catalogue

    def read_catalogue(self):

        return self._read_catalogue(self.filename)

    def get_zoom_from_catalogue(self, **kwargs):

        return self._get_zoom_from_catalogue(self.filename, **kwargs)


