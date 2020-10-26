import matplotlib

matplotlib.use('Agg')

import numpy as np
import unyt
import h5py
from typing import Tuple
import swiftsimio as sw
from swiftsimio.visualisation.projection import project_pixel_grid
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from vr_rendezvous import find_object

resolution = 2048


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str
    

def wrap(dx, box):
    result = dx
    index = np.where(dx > (0.5 * box))[0]
    result[index] -= box
    index = np.where(dx < (-0.5 * box))[0]
    result[index] += box
    return result


def dm_render(swio_data, region: list = None, resolution: int = 1024):

    # Project the dark matter mass
    dm_map = project_pixel_grid(
        # Note here that we pass in the dark matter dataset not the whole
        # data object, to specify what particle type we wish to visualise
        data=swio_data.dark_matter,
        boxsize=swio_data.metadata.boxsize,
        resolution=resolution,
        project=None,
        parallel=True,
        region=region
    )
    return dm_map


def dm_map_parent(
        run_name: str,
        velociraptor_properties_parent: str,
        snap_filepath_parent: str,
        velociraptor_properties_zoom: str,
        out_to_radius: Tuple[int, str] = (5, 'R200c'),
        highres_radius: Tuple[int, str] = (6, 'R500c'),
        output_directory: str = '.'
) -> None:
    print(f"Rendering {snap_filepath_parent}...")

    # Rendezvous over parent VR catalogue using zoom information
    with h5py.File(velociraptor_properties_zoom, 'r') as vr_file:
        idx, M200c, R200c, Xcminpot, Ycminpot, Zcminpot = find_object(
            vr_properties_catalog=velociraptor_properties_parent,
            sample_mass_lower_lim=vr_file['/Mass_200crit'][0] * 1e10 * 0.9,
            sample_x=vr_file['/Xcminpot'][0],
            sample_y=vr_file['/Ycminpot'][0],
            sample_z=vr_file['/Zcminpot'][0],
        )
    with h5py.File(velociraptor_properties_parent, 'r') as vr_file:
        R500c = unyt.unyt_quantity(vr_file['/SO_R_500_rhocrit'][idx], unyt.Mpc)

    M200c = unyt.unyt_quantity(M200c, unyt.Solar_Mass)
    R200c = unyt.unyt_quantity(R200c, unyt.Mpc)
    xCen = unyt.unyt_quantity(Xcminpot, unyt.Mpc)
    yCen = unyt.unyt_quantity(Ycminpot, unyt.Mpc)
    zCen = unyt.unyt_quantity(Zcminpot, unyt.Mpc)

    if out_to_radius[1] == 'R200c':
        size = out_to_radius[0] * R200c
    elif out_to_radius[1] == 'R500c':
        size = out_to_radius[0] * R500c
    elif out_to_radius[1] == 'Mpc' or out_to_radius[1] is None:
        size = unyt.unyt_quantity(out_to_radius[0], unyt.Mpc)

    if highres_radius[1] == 'R200c':
        _highres_radius = highres_radius[0] * R200c
    elif highres_radius[1] == 'R500c':
        _highres_radius = highres_radius[0] * R500c
    elif highres_radius[1] == 'Mpc' or highres_radius[1] is None:
        _highres_radius = unyt.unyt_quantity(highres_radius[0], unyt.Mpc)

    # Construct spatial mask to feed into swiftsimio
    mask = sw.mask(snap_filepath_parent)
    region = [
        [xCen - size, xCen + size],
        [yCen - size, yCen + size],
        [zCen - size, zCen + size]
    ]
    mask.constrain_spatial(region)
    data = sw.load(snap_filepath_parent, mask=mask)

    # Generate smoothing lengths for the dark matter
    data.dark_matter.smoothing_lengths = generate_smoothing_lengths(
        data.dark_matter.coordinates,
        data.metadata.boxsize,
        kernel_gamma=1.8,
        neighbours=57,
        speedup_fac=2,
        dimension=3,
    )

    data.dark_matter.coordinates[:, 0] = wrap(
        data.dark_matter.coordinates[:, 0] - xCen,
        data.metadata.boxsize[0]
    )
    data.dark_matter.coordinates[:, 1] = wrap(
        data.dark_matter.coordinates[:, 1] - yCen,
        data.metadata.boxsize[1]
    )
    data.dark_matter.coordinates[:, 2] = wrap(
        data.dark_matter.coordinates[:, 2] - zCen,
        data.metadata.boxsize[2]
    )
    
    dm_mass = dm_render(data, region=(
        [
            xCen - size,
            xCen + size,
            yCen - size,
            yCen + size
        ]
    ), resolution=resolution)

    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=resolution // 8)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.imshow(dm_mass.T, norm=LogNorm(), cmap="inferno", origin="lower", extent=(region[0] + region[1]))
    info = ax.text(
        0.025,
        0.025,
        (
            f"Halo {run_name:s} DMO - parent\n"
            f"$z={data.metadata.z:3.3f}$\n"
            f"$M_{{200c}}={latex_float(M200c.value)}\\ {M200c.units.latex_repr}$\n"
            f"$R_{{200c}}={latex_float(R200c.value)}\\ {R200c.units.latex_repr}$\n"
            f"$R_\\mathrm{{clean}}={highres_radius[0]}\\ {highres_radius[1]}$"
        ),
        color="white",
        ha="left",
        va="bottom",
        alpha=0.8,
        transform=ax.transAxes,
    )
    info.set_bbox(dict(facecolor='black', alpha=0.2, edgecolor='grey'))
    ax.text(
        xCen,
        yCen + 1.05 * R200c,
        r"$R_{200c}$",
        color="black",
        ha="center",
        va="bottom"
    )
    ax.text(
        xCen,
        yCen + 1.02 * _highres_radius,
        r"$R_\mathrm{clean}$",
        color="white",
        ha="center",
        va="bottom"
    )
    circle_r200 = plt.Circle((xCen, yCen), R200c, color="black", fill=False, linestyle='-')
    circle_clean = plt.Circle((xCen, yCen), _highres_radius.value, color="white", fill=False, linestyle=':')
    ax.add_artist(circle_r200)
    ax.add_artist(circle_clean)
    ax.set_xlim([xCen.value - size.value, xCen.value + size.value])
    ax.set_ylim([yCen.value - size.value, yCen.value + size.value])
    fig.savefig(f"{output_directory}/{run_name}_dark_matter_map_parent.png")
    plt.close(fig)
    print(f"Saved: {output_directory}/{run_name}_dark_matter_map_parent.png")
    del data, dm_mass

    return


def dm_map_zoom(
        run_name: str,
        snap_filepath_zoom: str,
        velociraptor_properties_zoom: str,
        out_to_radius: Tuple[int, str] = (5, 'R200c'),
        highres_radius: Tuple[int, str] = (6, 'R500c'),
        output_directory: str = '.'
) -> None:
    print(f"Rendering {snap_filepath_zoom}...")

    # Rendezvous over parent VR catalogue using zoom information
    with h5py.File(velociraptor_properties_zoom, 'r') as vr_file:

        M200c = unyt.unyt_quantity(vr_file['/Mass_200crit'][0] * 1e10, unyt.Solar_Mass)
        R200c = unyt.unyt_quantity(vr_file['/R_200crit'][0], unyt.Mpc)
        R500c = unyt.unyt_quantity(vr_file['/SO_R_500_rhocrit'][0], unyt.Mpc)
        xCen = unyt.unyt_quantity(vr_file['/Xcminpot'][0], unyt.Mpc)
        yCen = unyt.unyt_quantity(vr_file['/Ycminpot'][0], unyt.Mpc)
        zCen = unyt.unyt_quantity(vr_file['/Zcminpot'][0], unyt.Mpc)

    if out_to_radius[1] == 'R200c':
        size = out_to_radius[0] * R200c
    elif out_to_radius[1] == 'R500c':
        size = out_to_radius[0] * R500c
    elif out_to_radius[1] == 'Mpc' or out_to_radius[1] is None:
        size = unyt.unyt_quantity(out_to_radius[0], unyt.Mpc)

    if highres_radius[1] == 'R200c':
        _highres_radius = highres_radius[0] * R200c
    elif highres_radius[1] == 'R500c':
        _highres_radius = highres_radius[0] * R500c
    elif highres_radius[1] == 'Mpc' or highres_radius[1] is None:
        _highres_radius = unyt.unyt_quantity(highres_radius[0], unyt.Mpc)

    # Construct spatial mask to feed into swiftsimio
    mask = sw.mask(snap_filepath_zoom)
    region = [
        [xCen - size, xCen + size],
        [yCen - size, yCen + size],
        [zCen - size, zCen + size]
    ]
    mask.constrain_spatial(region)
    data = sw.load(snap_filepath_zoom, mask=mask)

    # Generate smoothing lengths for the dark matter
    data.dark_matter.smoothing_lengths = generate_smoothing_lengths(
        data.dark_matter.coordinates,
        data.metadata.boxsize,
        kernel_gamma=1.8,
        neighbours=57,
        speedup_fac=2,
        dimension=3,
    )

    data.dark_matter.coordinates[:, 0] = wrap(
        data.dark_matter.coordinates[:, 0] - xCen,
        data.metadata.boxsize[0]
    )
    data.dark_matter.coordinates[:, 1] = wrap(
        data.dark_matter.coordinates[:, 1] - yCen,
        data.metadata.boxsize[1]
    )
    data.dark_matter.coordinates[:, 2] = wrap(
        data.dark_matter.coordinates[:, 2] - zCen,
        data.metadata.boxsize[2]
    )
    dm_mass = dm_render(data, region=(
        [
            xCen - size,
            xCen + size,
            yCen - size,
            yCen + size
        ]
    ), resolution=resolution)

    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=resolution // 8)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.imshow(dm_mass.T, norm=LogNorm(), cmap="inferno", origin="lower", extent=(region[0] + region[1]))
    info = ax.text(
        0.025,
        0.025,
        (
            f"Halo {run_name:s} DMO - zoom\n"
            f"$z={data.metadata.z:3.3f}$\n"
            f"$M_{{200c}}={latex_float(M200c.value)}\\ {M200c.units.latex_repr}$\n"
            f"$R_{{200c}}={latex_float(R200c.value)}\\ {R200c.units.latex_repr}$\n"
            f"$R_\\mathrm{{clean}}={highres_radius[0]}\\ {highres_radius[1]}$"
        ),
        color="white",
        ha="left",
        va="bottom",
        alpha=0.8,
        transform=ax.transAxes,
    )
    info.set_bbox(dict(facecolor='black', alpha=0.2, edgecolor='grey'))
    ax.text(
        xCen,
        yCen + 1.05 * R200c,
        r"$R_{200c}$",
        color="black",
        ha="center",
        va="bottom"
    )
    ax.text(
        xCen,
        yCen + 1.02 * _highres_radius,
        r"$R_\mathrm{clean}$",
        color="white",
        ha="center",
        va="bottom"
    )
    circle_r200 = plt.Circle((xCen, yCen), R200c, color="black", fill=False, linestyle='-')
    circle_clean = plt.Circle((xCen, yCen), _highres_radius.value, color="white", fill=False, linestyle=':')
    ax.add_artist(circle_r200)
    ax.add_artist(circle_clean)
    ax.set_xlim([xCen.value - size.value, xCen.value + size.value])
    ax.set_ylim([yCen.value - size.value, yCen.value + size.value])
    fig.savefig(f"{output_directory}/{run_name}_dark_matter_map_zoom.png")
    plt.close(fig)
    print(f"Saved: {output_directory}/{run_name}_dark_matter_map_zoom.png")
    del data, dm_mass

    return


if __name__ == "__main__":

    run_name = 'L0300N0564_VR93'

    snap_filepath_parent = (
        "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
        "EAGLE-XL_L0300N0564_DMONLY_0036.hdf5"
    )

    velociraptor_properties_parent = (
        "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
        "stf_swiftdm_3dfof_subhalo_0036/"
        "stf_swiftdm_3dfof_subhalo_0036.VELOCIraptor.properties.0"
    )

    snap_filepath_zoom = "/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo/L0300N0564_VR93/snapshots/L0300N0564_VR93_0199.hdf5"
    velociraptor_properties_zoom = "/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo/L0300N0564_VR93/properties"

    output_directory = "/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"
    out_to_radius = (5, 'R200c')
    highres_radius = (6, 'R500c')

    dm_map_parent(
        run_name=run_name,
        velociraptor_properties_parent=velociraptor_properties_parent,
        snap_filepath_parent=snap_filepath_parent,
        velociraptor_properties_zoom=velociraptor_properties_zoom,
        out_to_radius=out_to_radius,
        highres_radius=highres_radius,
        output_directory=output_directory,
    )

    dm_map_zoom(
        run_name=run_name,
        snap_filepath_zoom=snap_filepath_zoom,
        velociraptor_properties_zoom=velociraptor_properties_zoom,
        out_to_radius=out_to_radius,
        highres_radius=highres_radius,
        output_directory=output_directory,
    )
