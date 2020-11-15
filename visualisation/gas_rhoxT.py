import matplotlib

matplotlib.use('Agg')

import numpy as np
import unyt
import h5py
from typing import Tuple
import swiftsimio as sw
from swiftsimio.visualisation.projection import project_pixel_grid
from swiftsimio.visualisation.sphviewer import SPHViewerWrapper
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


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


def gas_render(swio_data, region: list = None, resolution: int = 1024):

    # Project the dark matter mass
    dm_map = project_pixel_grid(
        # Note here that we pass in the dark matter dataset not the whole
        # data object, to specify what particle type we wish to visualise
        data=swio_data.gas,
        boxsize=swio_data.metadata.boxsize,
        resolution=resolution,
        project='densities',
        parallel=True,
        region=region
    )
    return dm_map


def gas_map_zoom(
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
    gas_rho = SPHViewerWrapper(data.gas, smooth_over="densities")
    gas_rho.quick_view(xsize=resolution, ysize=resolution, r="infinity")

    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=resolution // 8)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.imshow(gas_rho.image.T, norm=LogNorm(), cmap="twilight", origin="lower", extent=(region[0] + region[1]))
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
    plt.close('all')

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

    gas_map_zoom(
        run_name=run_name,
        snap_filepath_zoom=snap_filepath_zoom,
        velociraptor_properties_zoom=velociraptor_properties_zoom,
        out_to_radius=out_to_radius,
        highres_radius=highres_radius,
        output_directory=output_directory,
    )
