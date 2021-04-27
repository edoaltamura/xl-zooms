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
            map_extent_radius: unyt_quantity = 5 * kpc,
            **kwargs
    ):
        sw_data, vr_data = self.get_handles_from_zoom(zoom_obj, path_to_snap, path_to_catalogue, mask_radius_r500=1, **kwargs)

        m500 = vr_data.spherical_overdensities.mass_500_rhocrit[0].to('Msun')
        r500 = vr_data.spherical_overdensities.r_500_rhocrit[0].to('Mpc')
        xcminpot = vr_data.positions.xcminpot[0].to('Mpc')
        ycminpot = vr_data.positions.ycminpot[0].to('Mpc')
        zcminpot = vr_data.positions.zcminpot[0].to('Mpc')

        mapsize = map_extent_radius.to('Mpc')

        sw_data.black_holes.radial_distances.convert_to_physical()
        sw_data.black_holes.coordinates.convert_to_physical()
        sw_data.black_holes.subgrid_masses.convert_to_physical()
        sw_data.black_holes.smoothing_lengths.convert_to_physical()

        sw_data.gas.radial_distances.convert_to_physical()
        sw_data.gas.coordinates.convert_to_physical()
        sw_data.gas.masses.convert_to_physical()

        mask_bh = np.where(sw_data.black_holes.radial_distances <= mapsize)[0]
        bh_coord = sw_data.black_holes.coordinates[mask_bh].value
        bh_coord[:, 0] -= xcminpot.v
        bh_coord[:, 1] -= ycminpot.v
        bh_coord[:, 2] -= zcminpot.v
        bh_mass = sw_data.black_holes.subgrid_masses[mask_bh]
        bh_mass_scaled = (bh_mass - bh_mass.min()) / (bh_mass.max() - bh_mass.min())
        bh_smoothing_lenghts = sw_data.black_holes.smoothing_lengths[mask_bh]

        # Get the central BH closest to centre of halo
        central_bh_index = np.argmin(sw_data.black_holes.radial_distances[mask_bh])

        print(f"Plotting {len(mask_bh):d} BHs")

        # Get gas particles close to the BH
        # Select hot gas within sphere
        # mask_gas = np.where(
        #     (sw_data.gas.radial_distances <= mapsize) &
        #     (sw_data.gas.temperatures > Tcut_halogas) &
        #     (sw_data.gas.fofgroup_ids == 1)
        # )[0]
        mask_gas = np.where(sw_data.gas.radial_distances <= mapsize)[0]

        gas_coord = sw_data.gas.coordinates[mask_gas].value
        gas_coord[:, 0] -= xcminpot.v
        gas_coord[:, 1] -= ycminpot.v
        gas_coord[:, 2] -= zcminpot.v
        gas_mass = sw_data.gas.masses[mask_gas]
        gas_mass_scaled = (gas_mass - gas_mass.min()) / (gas_mass.max() - gas_mass.min())
        print(f"Plotting {len(mask_gas):d} gas particles")
        print(f"Gas mass: max {gas_mass.max().to('Msun'):.2E}, min {gas_mass.min().to('Msun'):.2E}")

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
            s=[10 * 4 ** n for n in gas_mass_scaled],
            marker='.', edgecolors='none'
        )

        kwargs_bh = dict(
            color='k',
            marker='*',
            edgecolors='none',
            s=[10*4**n for n in bh_mass_scaled]
        )

        axes[0, 0].scatter(gas_coord[:, 0], gas_coord[:, 1], **kwargs_gas)
        axes[0, 0].scatter(bh_coord[:, 0], bh_coord[:, 1], **kwargs_bh)
        axes[0, 0].scatter(bh_coord[central_bh_index, 0], bh_coord[central_bh_index, 1], color='brown', marker='*', edgecolors='none', s=10*4**bh_mass_scaled[central_bh_index])
        axes[0, 0].add_patch(plt.Circle((bh_coord[central_bh_index, 0], bh_coord[central_bh_index, 1]), bh_smoothing_lenghts[central_bh_index], facecolor='brown', alpha=0.2, edgecolor='none'))

        axes[0, 0].axhline(y=0, linestyle='--', linewidth=1, color='grey')
        axes[0, 0].axvline(x=0, linestyle='--', linewidth=1, color='grey')
        axes[0, 0].set_xlim([-mapsize, mapsize])
        axes[0, 0].set_ylim([-mapsize, mapsize])
        axes[0, 0].set_aspect('equal')
        axes[0, 0].set_ylabel('y [Mpc]')

        axes[0, 1].scatter(gas_coord[:, 2], gas_coord[:, 1], **kwargs_gas)
        axes[0, 1].scatter(bh_coord[:, 2], bh_coord[:, 1], **kwargs_bh)
        axes[0, 1].scatter(bh_coord[central_bh_index, 2], bh_coord[central_bh_index, 1], color='brown', marker='*', edgecolors='none', s=10*4**bh_mass_scaled[central_bh_index])
        axes[0, 1].add_patch(plt.Circle((bh_coord[central_bh_index, 2], bh_coord[central_bh_index, 1]), bh_smoothing_lenghts[central_bh_index], facecolor='brown', alpha=0.2, edgecolor='none'))

        axes[0, 1].axhline(y=0, linestyle='--', linewidth=1, color='grey')
        axes[0, 1].axvline(x=0, linestyle='--', linewidth=1, color='grey')
        axes[0, 1].set_xlim([-mapsize, mapsize])
        axes[0, 1].set_ylim([-mapsize, mapsize])
        axes[0, 1].set_aspect('equal')
        axes[0, 1].set_xlabel('z [Mpc]')

        axes[1, 0].scatter(gas_coord[:, 0], gas_coord[:, 2], **kwargs_gas)
        axes[1, 0].scatter(bh_coord[:, 0], bh_coord[:, 2], **kwargs_bh)
        axes[1, 0].scatter(bh_coord[central_bh_index, 0], bh_coord[central_bh_index, 2], color='brown', marker='*', edgecolors='none', s=10*4**bh_mass_scaled[central_bh_index])
        axes[1, 0].add_patch(plt.Circle((bh_coord[central_bh_index, 0], bh_coord[central_bh_index, 2]), bh_smoothing_lenghts[central_bh_index], facecolor='brown', alpha=0.2, edgecolor='none'))

        axes[1, 0].axhline(y=0, linestyle='--', linewidth=1, color='grey')
        axes[1, 0].axvline(x=0, linestyle='--', linewidth=1, color='grey')
        axes[1, 0].set_xlim([-mapsize, mapsize])
        axes[1, 0].set_ylim([-mapsize, mapsize])
        axes[1, 0].set_aspect('equal')
        axes[1, 0].set_xlabel('x [Mpc]')
        axes[1, 0].set_ylabel('z [Mpc]')

        axes[1, 1].remove()

        plt.show()
