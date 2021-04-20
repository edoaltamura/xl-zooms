import numpy as np
import swiftsimio
import velociraptor
from scipy.spatial import distance

from register import (
    args,
    DataframePickler,
    Zoom
)


class HaloProperty(object):

    def __init__(self):
        pass

    @staticmethod
    def get_handles_from_paths(
            path_to_snap: str,
            path_to_catalogue: str,
            mask_radius_r500: float = 6
    ) -> tuple:
        # Read in halo properties
        vr_handle = velociraptor.load(path_to_catalogue)
        a = vr_handle.a
        r500 = vr_handle.spherical_overdensities.r_500_rhocrit[0].to('Mpc')
        xcminpot = vr_handle.positions.xcminpot[0].to('Mpc')
        ycminpot = vr_handle.positions.ycminpot[0].to('Mpc')
        zcminpot = vr_handle.positions.zcminpot[0].to('Mpc')

        # Apply spatial mask to particles. SWIFTsimIO needs comoving coordinates
        # to filter particle coordinates, while VR outputs are in physical units.
        # Convert the region bounds to comoving, but keep the CoP and Rcrit in
        # physical units for later use.
        mask = swiftsimio.mask(path_to_snap, spatial_only=True)
        mask_radius = mask_radius_r500 * r500
        region = [
            [(xcminpot - mask_radius) / a, (xcminpot + mask_radius) / a],
            [(ycminpot - mask_radius) / a, (ycminpot + mask_radius) / a],
            [(zcminpot - mask_radius) / a, (zcminpot + mask_radius) / a]
        ]
        mask.constrain_spatial(region)
        sw_handle = swiftsimio.load(path_to_snap, mask=mask)

        # Convert datasets to physical quantities
        # R500c is already in physical units
        sw_handle.gas.coordinates.convert_to_physical()
        sw_handle.gas.masses.convert_to_physical()
        sw_handle.gas.temperatures.convert_to_physical()
        sw_handle.gas.densities.convert_to_physical()
        sw_handle.gas.entropies.convert_to_physical()
        sw_handle.gas.velocities.convert_to_physical()

        sw_handle.dark_matter.coordinates.convert_to_physical()
        sw_handle.dark_matter.masses.convert_to_physical()
        sw_handle.dark_matter.velocities.convert_to_physical()

        sw_handle.stars.coordinates.convert_to_physical()
        sw_handle.stars.masses.convert_to_physical()
        sw_handle.stars.velocities.convert_to_physical()

        sw_handle.black_holes.coordinates.convert_to_physical()
        sw_handle.black_holes.masses.convert_to_physical()
        sw_handle.black_holes.velocities.convert_to_physical()

        # If the mask overlaps with the box boundaries, wrap coordinates.
        boxsize = sw_handle.metadata.boxsize[0]
        centre_coordinates = np.array([xcminpot, ycminpot, zcminpot], dtype=np.float64)

        sw_handle.gas.coordinates = np.mod(
            sw_handle.gas.coordinates - centre_coordinates + 0.5 * boxsize,
            boxsize
        ) + centre_coordinates - 0.5 * boxsize

        sw_handle.dark_matter.coordinates = np.mod(
            sw_handle.dark_matter.coordinates - centre_coordinates + 0.5 * boxsize,
            boxsize
        ) + centre_coordinates - 0.5 * boxsize

        sw_handle.stars.coordinates = np.mod(
            sw_handle.stars.coordinates - centre_coordinates + 0.5 * boxsize,
            boxsize
        ) + centre_coordinates - 0.5 * boxsize

        sw_handle.black_holes.coordinates = np.mod(
            sw_handle.black_holes.coordinates - centre_coordinates + 0.5 * boxsize,
            boxsize
        ) + centre_coordinates - 0.5 * boxsize

        # Compute radial distances
        sw_handle.gas.radial_distances = distance.cdist(
            sw_handle.gas.coordinates,
            centre_coordinates.reshape(1, 3),
            metric='euclidean'
        ).reshape(len(sw_handle.gas.coordinates), )

        sw_handle.dark_matter.radial_distances = distance.cdist(
            sw_handle.dark_matter.coordinates,
            centre_coordinates.reshape(1, 3),
            metric='euclidean'
        ).reshape(len(sw_handle.dark_matter.coordinates), )

        sw_handle.stars.radial_distances = distance.cdist(
            sw_handle.stars.coordinates,
            centre_coordinates.reshape(1, 3),
            metric='euclidean'
        ).reshape(len(sw_handle.stars.coordinates), )

        sw_handle.black_holes.radial_distances = distance.cdist(
            sw_handle.black_holes.coordinates,
            centre_coordinates.reshape(1, 3),
            metric='euclidean'
        ).reshape(len(sw_handle.black_holes.coordinates), )

        return sw_handle, vr_handle

    def get_handles_from_zoom(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None,
            **kwargs
    ):
        assert (
                zoom_obj is not None or
                (path_to_snap is not None and path_to_snap is not None)
        ), (
            "Either a `Zoom` object must be specified or the absolute "
            "paths to the snapshot and properties catalogue files."
        )

        snapshot_file = path_to_snap
        catalog_file = path_to_catalogue

        if zoom_obj is not None:
            zoom_at_redshift = zoom_obj.get_redshift(args.redshift_index)
            snapshot_file = zoom_at_redshift.snapshot_path
            catalog_file = zoom_at_redshift.catalogue_properties_path

        return self.get_handles_from_paths(snapshot_file, catalog_file, **kwargs)

    def read_from_file(self):
        pickler = DataframePickler(self.storage_file)
        return pickler.load_from_pickle()
