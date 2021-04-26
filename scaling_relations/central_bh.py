import os.path
import numpy as np
from warnings import warn
from unyt import unyt_quantity, kpc, Mpc
from matplotlib import pyplot as plt
import matplotlib.colors as colors

from .halo_property import HaloProperty
from register import Zoom, Tcut_halogas, default_output_directory, args


class CentralBH(HaloProperty):

    def __init__(self):
        super().__init__()

    def process_single_halo(
            self,
            zoom_obj: Zoom = None,
            path_to_snap: str = None,
            path_to_catalogue: str = None,
            map_extent_radius: unyt_quantity = 10 * kpc,
            **kwargs
    ):
        sw_data, vr_data = self.get_handles_from_zoom(zoom_obj, path_to_snap, path_to_catalogue, mask_radius_r500=0.1, **kwargs)

        m500 = vr_data.spherical_overdensities.mass_500_rhocrit[0].to('Msun')
        r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc')
        xcminpot = vr_data.positions.xcminpot[0].to('Mpc')
        ycminpot = vr_data.positions.ycminpot[0].to('Mpc')
        zcminpot = vr_data.positions.zcminpot[0].to('Mpc')

        mapsize = map_extent_radius.to('Mpc')

        sw_data.black_holes.radial_distances.convert_to_physical()
        sw_data.black_holes.coordinates.convert_to_physical()
        sw_data.black_holes.subgrid_masses.convert_to_physical()

        sw_data.gas.radial_distances.convert_to_physical()
        sw_data.gas.coordinates.convert_to_physical()
        sw_data.gas.masses.convert_to_physical()

        # Get the central BH closest to centre of halo
        central_bh_index = np.argmin(sw_data.black_holes.radial_distances)
        central_bh_id_target = sw_data.black_holes.particle_ids[central_bh_index]
        central_bh_index = np.where(sw_data.black_holes.particle_ids == central_bh_id_target)[0]

        sw_data.gas.coordinates[:, 0] -= xcminpot
        sw_data.gas.coordinates[:, 1] -= ycminpot
        sw_data.gas.coordinates[:, 2] -= zcminpot
        sw_data.black_holes.coordinates[:, 0] -= xcminpot
        sw_data.black_holes.coordinates[:, 1] -= ycminpot
        sw_data.black_holes.coordinates[:, 2] -= zcminpot

        mask_bh = np.where(sw_data.black_holes.radial_distances <= mapsize)[0]

        print(f"Plotting {len(mask_bh):d} BHs")

        # Get gas particles close to the BH
        # Select hot gas within sphere
        mask_gas = np.where(
            (sw_data.gas.radial_distances <= mapsize) &
            (sw_data.gas.temperatures > Tcut_halogas) &
            (sw_data.gas.fofgroup_ids == 1)
        )[0]
        print(f"Plotting {len(mask_gas):d} gas particles")

        fig = plt.figure(figsize=(3, 3))
        gs = fig.add_gridspec(2, 2, hspace=0., wspace=0.)
        axes = gs.subplots(sharex=True, sharey=True)

        kwargs_gas = dict(
            c=sw_data.gas.temperatures[mask_gas],
            cmap='coolwarm',
            norm=colors.LogNorm(
                vmin=sw_data.gas.temperatures[mask_gas].min(),
                vmax=sw_data.gas.temperatures[mask_gas].max()
            ),
            marker='.', edgecolors='none'
        )
        kwargs_bh = dict(color='k', marker='*', edgecolors='none')

        axes[0, 0].scatter(sw_data.gas.coordinates[mask_gas, 0], sw_data.gas.coordinates[mask_gas, 1], **kwargs_gas)
        axes[0, 0].scatter(sw_data.black_holes.coordinates[mask_bh, 0], sw_data.black_holes.coordinates[mask_bh, 1], **kwargs_bh)
        axes[0, 0].scatter(sw_data.black_holes.coordinates[central_bh_index, 0], sw_data.black_holes.coordinates[central_bh_index, 1], color='r', marker='*', edgecolors='none', s=20)
        axes[0, 0].axhline(y=0, linestyle='--', linewidth=1, color='grey')
        axes[0, 0].axvline(x=0, linestyle='--', linewidth=1, color='grey')
        axes[0, 0].set_xlim([-mapsize, mapsize])
        axes[0, 0].set_ylim([-mapsize, mapsize])
        axes[0, 0].set_aspect('equal')

        axes[0, 1].scatter(sw_data.gas.coordinates[mask_gas, 2], sw_data.gas.coordinates[mask_gas, 1], **kwargs_gas)
        axes[0, 1].scatter(sw_data.black_holes.coordinates[mask_bh, 2], sw_data.black_holes.coordinates[mask_bh, 1], **kwargs_bh)
        axes[0, 1].scatter(sw_data.black_holes.coordinates[central_bh_index, 2], sw_data.black_holes.coordinates[central_bh_index, 1], color='r', marker='*', edgecolors='none', s=20)
        axes[0, 1].axhline(y=0, linestyle='--', linewidth=1, color='grey')
        axes[0, 1].axvline(x=0, linestyle='--', linewidth=1, color='grey')
        axes[0, 1].set_xlim([-mapsize, mapsize])
        axes[0, 1].set_ylim([-mapsize, mapsize])
        axes[0, 1].set_aspect('equal')

        axes[1, 0].scatter(sw_data.gas.coordinates[mask_gas, 0], sw_data.gas.coordinates[mask_gas, 2], **kwargs_gas)
        axes[1, 0].scatter(sw_data.black_holes.coordinates[mask_bh, 0], sw_data.black_holes.coordinates[mask_bh, 2], **kwargs_bh)
        axes[1, 0].scatter(sw_data.black_holes.coordinates[central_bh_index, 0], sw_data.black_holes.coordinates[central_bh_index, 2], color='r', marker='*', edgecolors='none', s=20)
        axes[1, 0].axhline(y=0, linestyle='--', linewidth=1, color='grey')
        axes[1, 0].axvline(x=0, linestyle='--', linewidth=1, color='grey')
        axes[1, 0].set_xlim([-mapsize, mapsize])
        axes[1, 0].set_ylim([-mapsize, mapsize])
        axes[1, 0].set_aspect('equal')

        axes[1, 1].remove()

        plt.show()
