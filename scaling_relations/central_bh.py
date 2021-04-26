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
            map_extent_radius: unyt_quantity = 50 * kpc,
            **kwargs
    ):
        sw_data, vr_data = self.get_handles_from_zoom(zoom_obj, path_to_snap, path_to_catalogue, **kwargs)

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

        print(sw_data.black_holes.coordinates)

        # Get the central BH closest to centre of halo
        central_bh_index = np.argmin(sw_data.black_holes.radial_distances)
        mask_bh = np.where(sw_data.black_holes.radial_distances <= mapsize)[0]
        bh_coord = sw_data.black_holes.coordinates[mask_bh].value
        bh_coord[:, 0] -= xcminpot
        bh_coord[:, 1] -= ycminpot
        bh_coord[:, 2] -= zcminpot

        print(f"Plotting {len(mask_bh):d} BHs", bh_coord)

        # Get gas particles close to the BH
        # Select hot gas within sphere
        mask_gas = np.where(
            (sw_data.gas.radial_distances <= mapsize) &
            (sw_data.gas.temperatures > Tcut_halogas) &
            (sw_data.gas.fofgroup_ids == 1)
        )[0]

        gas_coord = sw_data.gas.coordinates[mask_gas].value
        gas_coord[:, 0] -= xcminpot
        gas_coord[:, 1] -= ycminpot
        gas_coord[:, 2] -= zcminpot
        print(f"Plotting {len(mask_gas):d} gas particles", gas_coord)

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

        axes[0, 0].scatter(gas_coord[:, 0], gas_coord[:, 1], **kwargs_gas)
        axes[0, 0].scatter(bh_coord[:, 0], bh_coord[:, 1], **kwargs_bh)
        # axes[0, 0].scatter(bh_coord[central_bh_index, 0], bh_coord[central_bh_index, 1], color='r', marker='*', edgecolors='none', s=20)
        axes[0, 0].axhline(y=0, linestyle='--', linewidth=1, color='grey')
        axes[0, 0].axvline(x=0, linestyle='--', linewidth=1, color='grey')
        axes[0, 0].set_xlim([-mapsize, mapsize])
        axes[0, 0].set_ylim([-mapsize, mapsize])
        axes[0, 0].set_aspect('equal')

        axes[0, 1].scatter(gas_coord[:, 2], gas_coord[:, 1], **kwargs_gas)
        axes[0, 1].scatter(bh_coord[:, 2], bh_coord[:, 1], **kwargs_bh)
        # axes[0, 1].scatter(bh_coord[central_bh_index, 2], bh_coord[central_bh_index, 1], color='r', marker='*', edgecolors='none', s=20)
        axes[0, 1].axhline(y=0, linestyle='--', linewidth=1, color='grey')
        axes[0, 1].axvline(x=0, linestyle='--', linewidth=1, color='grey')
        axes[0, 1].set_xlim([-mapsize, mapsize])
        axes[0, 1].set_ylim([-mapsize, mapsize])
        axes[0, 1].set_aspect('equal')

        axes[1, 0].scatter(gas_coord[:, 0], gas_coord[:, 2], **kwargs_gas)
        axes[1, 0].scatter(bh_coord[:, 0], bh_coord[:, 2], **kwargs_bh)
        # axes[1, 0].scatter(bh_coord[central_bh_index, 0], bh_coord[central_bh_index, 2], color='r', marker='*', edgecolors='none', s=20)
        axes[1, 0].axhline(y=0, linestyle='--', linewidth=1, color='grey')
        axes[1, 0].axvline(x=0, linestyle='--', linewidth=1, color='grey')
        axes[1, 0].set_xlim([-mapsize, mapsize])
        axes[1, 0].set_ylim([-mapsize, mapsize])
        axes[1, 0].set_aspect('equal')

        axes[1, 1].remove()

        plt.show()
