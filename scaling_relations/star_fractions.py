import os.path
from warnings import warn

from .halo_property import HaloProperty
from register import Zoom, default_output_directory, args


class StarFractions(HaloProperty):

    def __init__(self):
        super().__init__()

        self.labels = ['f_star']

        self.filename = os.path.join(
            default_output_directory,
            'intermediate',
            f'star_fractions_{args.redshift_index:04d}.pkl'
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
            **kwargs
    ):
        catalogue_file = os.path.join(
            default_output_directory,
            'intermediate',
            f'vrproperties_{args.redshift_index:04d}.pkl'
        )
        data = self._get_zoom_from_catalogue(catalogue_file, zoom_obj=zoom_obj, **kwargs)

        m500 = data['m500']
        m_star500c = data['m_star500c']
        star_fraction = m_star500c / m500

        self.check_value(star_fraction)
        return star_fraction

    def process_catalogue(self):

        catalogue = self._process_catalogue(self.process_single_halo, labels=self.labels)
        self.dump_to_pickle(self.filename, catalogue)

    def read_catalogue(self):

        return self._read_catalogue(self.filename)

    def get_zoom_from_catalogue(self, **kwargs):

        return self._get_zoom_from_catalogue(self.filename, **kwargs)
