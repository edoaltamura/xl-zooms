import matplotlib

matplotlib.use('Agg')

import numpy as np
import unyt
import h5py
from typing import Tuple
import matplotlib.pyplot as plt
import swiftsimio as sw

try:
    plt.style.use("mnras.mplstyle")
except:
    pass

resolution = 2048


def wrap(dx, box):
    result = dx
    index = np.where(dx > (0.5 * box))[0]
    result[index] -= box
    index = np.where(dx < (-0.5 * box))[0]
    result[index] += box
    return result


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def contamination_map(
        run_name: str,
        velociraptor_properties_zoom: str,
        snap_filepath_zoom: str,
        out_to_radius: Tuple[int, str] = (5, 'R200c'),
        highres_radius: Tuple[int, str] = (6, 'R500c'),
        output_directory: str = '.'
) -> None:
    # Rendezvous over parent VR catalogue using zoom information
    with h5py.File(velociraptor_properties_zoom, 'r') as vr_file:
        M200c = unyt.unyt_quantity(vr_file['/Mass_200crit'][0] * 1e10, unyt.Solar_Mass)
        R200c = unyt.unyt_quantity(vr_file['/R_200crit'][0], unyt.Mpc)
        R500c = unyt.unyt_quantity(vr_file['/SO_R_500_rhocrit'][0], unyt.Mpc)
        xCen = unyt.unyt_quantity(vr_file['/Xcminpot'][0], unyt.Mpc)
        yCen = unyt.unyt_quantity(vr_file['/Ycminpot'][0], unyt.Mpc)
        zCen = unyt.unyt_quantity(vr_file['/Zcminpot'][0], unyt.Mpc)

    # EAGLE-XL data path
    print(f"Rendering {snap_filepath_zoom}...")

    if out_to_radius[1] == 'R200c':
        size = out_to_radius[0] * R200c
    elif out_to_radius[1] == 'R500c':
        size = out_to_radius[0] * R500c
    elif out_to_radius[1] == 'Mpc' or out_to_radius[1] is None:
        size = unyt.unyt_quantity(out_to_radius[0], unyt.Mpc)
    else:
        raise ValueError("The `out_to_radius` input is not in the correct format or not recognised.")

    if highres_radius[1] == 'R200c':
        _highres_radius = highres_radius[0] * R200c
    elif highres_radius[1] == 'R500c':
        _highres_radius = highres_radius[0] * R500c
    elif highres_radius[1] == 'Mpc' or highres_radius[1] is None:
        _highres_radius = unyt.unyt_quantity(highres_radius[0], unyt.Mpc)
    else:
        raise ValueError("The `highres_radius` input is not in the correct format or not recognised.")

    mask = sw.mask(snap_filepath_zoom)
    region = [
        [xCen - size, xCen + size],
        [yCen - size, yCen + size],
        [zCen - size, zCen + size]
    ]
    mask.constrain_spatial(region)

    # Load data using mask
    data = sw.load(snap_filepath_zoom, mask=mask)
    posDM = data.dark_matter.coordinates / data.metadata.a
    highres_coordinates = {
        'x': wrap(posDM[:, 0] - xCen, data.metadata.boxsize[0]),
        'y': wrap(posDM[:, 1] - yCen, data.metadata.boxsize[1]),
        'z': wrap(posDM[:, 2] - zCen, data.metadata.boxsize[2]),
        'r': np.sqrt(wrap(posDM[:, 0] - xCen, data.metadata.boxsize[0]) ** 2 +
                     wrap(posDM[:, 1] - yCen, data.metadata.boxsize[1]) ** 2 +
                     wrap(posDM[:, 2] - zCen, data.metadata.boxsize[2]) ** 2)
    }
    del posDM
    posDM = data.boundary.coordinates / data.metadata.a
    lowres_coordinates = {
        'x': wrap(posDM[:, 0] - xCen, data.metadata.boxsize[0]),
        'y': wrap(posDM[:, 1] - yCen, data.metadata.boxsize[1]),
        'z': wrap(posDM[:, 2] - zCen, data.metadata.boxsize[2]),
        'r': np.sqrt(wrap(posDM[:, 0] - xCen, data.metadata.boxsize[0]) ** 2 +
                     wrap(posDM[:, 1] - yCen, data.metadata.boxsize[1]) ** 2 +
                     wrap(posDM[:, 2] - zCen, data.metadata.boxsize[2]) ** 2)
    }
    del posDM

    # Flag contamination particles within 5 R200
    contaminated_idx = np.where(lowres_coordinates['r'] < _highres_radius)[0]
    contaminated_r200_idx = np.where(lowres_coordinates['r'] < 1. * R200c)[0]
    print(f"Total low-res DM: {len(lowres_coordinates['r'])} particles detected")
    print(f"Contaminating low-res DM (< R_clean): {len(contaminated_idx)} particles detected")
    print(f"Contaminating low-res DM (< R200c): {len(contaminated_r200_idx)} particles detected")

    # Make particle maps
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), dpi=resolution // 6)
    
    for ax in axs:
        ax.set_aspect('equal')
        
        ax.text(
            0.025,
            0.025,
            (
                f"Halo {run_name:s} DMO\n"
                f"$z={data.metadata.z:3.3f}$\n"
                f"$M_{{200c}}={latex_float(M200c.value)}$ M$_\odot$\n"
                f"$R_{{200c}}={latex_float(R200c.value)}$ Mpc\n"
                f"$R_{{\\rm clean}}={latex_float(_highres_radius.value)}$ Mpc"
            ),
            color="black",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
        )
        ax.text(
            0,
            0 + 1.05 * R200c,
            r"$R_{200c}$",
            color="black",
            ha="center",
            va="bottom"
        )
        ax.text(
            0,
            0 + 1.002 * 5 * R200c,
            r"$5 \times R_{200c}$",
            color="grey",
            ha="center",
            va="bottom"
        )
        ax.text(
            0,
            0 + 1.02 * _highres_radius,
            r"$R_\mathrm{clean}$",
            color="red",
            ha="center",
            va="bottom"
        )
        circle_r200 = plt.Circle((0, 0), R200c, color="black", fill=False, linestyle='-')
        circle_5r200 = plt.Circle((0, 0), 5 * R200c, color="grey", fill=False, linestyle='--')
        circle_clean = plt.Circle((0, 0), _highres_radius.value, color="red", fill=False, linestyle=':')
        ax.add_artist(circle_r200)
        ax.add_artist(circle_5r200)
        ax.add_artist(circle_clean)
        ax.set_xlim([-size.value, size.value])
        ax.set_ylim([-size.value, size.value])

    axs[0].plot(
        highres_coordinates['x'][::2],
        highres_coordinates['y'][::2],
        ',',
        c="C0",
        alpha=0.1,
        label='Highres'
    )
    axs[0].plot(
        lowres_coordinates['x'][contaminated_idx],
        lowres_coordinates['y'][contaminated_idx],
        'x',
        c="red",
        alpha=1,
        label='Lowres contaminating'
    )
    axs[0].plot(
        lowres_coordinates['x'][~contaminated_idx],
        lowres_coordinates['y'][~contaminated_idx],
        '.',
        c="green",
        alpha=0.2,
        label='Lowres clean'
    )
    axs[0].set_ylabel(r"$y$ [Mpc]")
    axs[0].set_xlabel(r"$x$ [Mpc]")

    axs[1].plot(
        highres_coordinates['y'][::2],
        highres_coordinates['z'][::2],
        ',',
        c="C0",
        alpha=0.1,
        label='Highres'
    )
    axs[1].plot(
        lowres_coordinates['y'][contaminated_idx],
        lowres_coordinates['z'][contaminated_idx],
        'x',
        c="red",
        alpha=1,
        label='Lowres contaminating'
    )
    axs[1].plot(
        lowres_coordinates['y'][~contaminated_idx],
        lowres_coordinates['z'][~contaminated_idx],
        '.',
        c="green",
        alpha=0.2,
        label='Lowres clean'
    )
    axs[1].set_ylabel(r"$z$ [Mpc]")
    axs[1].set_xlabel(r"$y$ [Mpc]")

    axs[2].plot(
        highres_coordinates['x'][::2],
        highres_coordinates['z'][::2],
        ',',
        c="C0",
        alpha=0.1,
        label='Highres'
    )
    axs[2].plot(
        lowres_coordinates['x'][contaminated_idx],
        lowres_coordinates['z'][contaminated_idx],
        'x',
        c="red",
        alpha=1,
        label='Lowres contaminating'
    )
    axs[2].plot(
        lowres_coordinates['x'][~contaminated_idx],
        lowres_coordinates['z'][~contaminated_idx],
        '.',
        c="green",
        alpha=0.2,
        label='Lowres clean'
    )
    axs[2].set_ylabel(r"$z$ [Mpc]")
    axs[2].set_xlabel(r"$x$ [Mpc]")
    
    plt.legend(loc="upper right")
    fig.savefig(f"{output_directory}/{run_name}_contamination_map{out_to_radius[0]}{out_to_radius[1]}.png")
    print(f"Saved: {output_directory}/{run_name}_contamination_map{out_to_radius[0]}{out_to_radius[1]}.png")
    plt.close(fig)
    plt.close('all')


def contamination_radial_histogram(
        run_name: str,
        velociraptor_properties_zoom: str,
        snap_filepath_zoom: str,
        out_to_radius: Tuple[int, str] = (5, 'R200c'),
        highres_radius: Tuple[int, str] = (6, 'R500c'),
        output_directory: str = '.'
) -> None:
    # Rendezvous over parent VR catalogue using zoom information
    with h5py.File(velociraptor_properties_zoom, 'r') as vr_file:
        M200c = unyt.unyt_quantity(vr_file['/Mass_200crit'][0] * 1e10, unyt.Solar_Mass)
        R200c = unyt.unyt_quantity(vr_file['/R_200crit'][0], unyt.Mpc)
        R500c = unyt.unyt_quantity(vr_file['/SO_R_500_rhocrit'][0], unyt.Mpc)
        xCen = unyt.unyt_quantity(vr_file['/Xcminpot'][0], unyt.Mpc)
        yCen = unyt.unyt_quantity(vr_file['/Ycminpot'][0], unyt.Mpc)
        zCen = unyt.unyt_quantity(vr_file['/Zcminpot'][0], unyt.Mpc)

    # EAGLE-XL data path
    print(f"Rendering {snap_filepath_zoom}...")

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

    mask = sw.mask(snap_filepath_zoom)
    region = [
        [xCen - size, xCen + size],
        [yCen - size, yCen + size],
        [zCen - size, zCen + size]
    ]
    mask.constrain_spatial(region)

    # Load data using mask
    data = sw.load(snap_filepath_zoom, mask=mask)
    posDM = data.dark_matter.coordinates / data.metadata.a
    highres_coordinates = {
        'x': wrap(posDM[:, 0] - xCen, data.metadata.boxsize[0]),
        'y': wrap(posDM[:, 1] - yCen, data.metadata.boxsize[1]),
        'z': wrap(posDM[:, 2] - zCen, data.metadata.boxsize[2]),
        'r': np.sqrt(wrap(posDM[:, 0] - xCen, data.metadata.boxsize[0]) ** 2 +
                     wrap(posDM[:, 1] - yCen, data.metadata.boxsize[1]) ** 2 +
                     wrap(posDM[:, 2] - zCen, data.metadata.boxsize[2]) ** 2)
    }
    del posDM
    posDM = data.boundary.coordinates / data.metadata.a
    lowres_coordinates = {
        'x': wrap(posDM[:, 0] - xCen, data.metadata.boxsize[0]),
        'y': wrap(posDM[:, 1] - yCen, data.metadata.boxsize[1]),
        'z': wrap(posDM[:, 2] - zCen, data.metadata.boxsize[2]),
        'r': np.sqrt(wrap(posDM[:, 0] - xCen, data.metadata.boxsize[0]) ** 2 +
                     wrap(posDM[:, 1] - yCen, data.metadata.boxsize[1]) ** 2 +
                     wrap(posDM[:, 2] - zCen, data.metadata.boxsize[2]) ** 2)
    }
    del posDM

    # Histograms
    contaminated_idx = np.where(lowres_coordinates['r'] < _highres_radius)[0]
    bins = np.linspace(0, size, 40)
    hist, bin_edges = np.histogram(lowres_coordinates['r'][contaminated_idx], bins=bins)
    lowres_coordinates['r_bins'] = (bin_edges[1:] + bin_edges[:-1]) / 2 / R200c
    lowres_coordinates['hist_contaminating'] = hist
    del hist, bin_edges
    hist, _ = np.histogram(lowres_coordinates['r'], bins=bins)
    lowres_coordinates['hist_all'] = hist
    del hist
    hist, _ = np.histogram(highres_coordinates['r'], bins=bins)
    highres_coordinates['r_bins'] = lowres_coordinates['r_bins']
    highres_coordinates['hist_all'] = hist
    del bins, hist

    # Make radial distribution plot
    fig, ax = plt.subplots(figsize=(5, 5), dpi=resolution // 5)

    ax.set_yscale('log')
    ax.set_ylabel("Number of particles")
    ax.set_xlabel(r"$R\ /\ R_{200c}$")

    ax.step(
        highres_coordinates['r_bins'],
        highres_coordinates['hist_all'],
        where='mid',
        color='grey',
        label='Highres all'
    )
    ax.step(
        lowres_coordinates['r_bins'],
        lowres_coordinates['hist_all'],
        where='mid',
        color='green',
        label='Lowres all'
    )
    ax.step(
        lowres_coordinates['r_bins'],
        lowres_coordinates['hist_contaminating'],
        where='mid',
        color='red',
        label='Lowres contaminating'
    )
    ax.text(
        0.025,
        0.025,
        (
            f"Halo {run_name:s} DMO\n"
            f"$z={data.metadata.z:3.3f}$\n"
            f"$M_{{200c}}={latex_float(M200c.value)}$ M$_\odot$\n"
            f"$R_{{200c}}={latex_float(R200c.value)}$ Mpc\n"
            f"$R_{{\\rm clean}}={latex_float(_highres_radius.value)}$ Mpc"
        ),
        color="black",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.axvline(1, color="grey", linestyle='--')
    ax.axvline(_highres_radius / R200c, color="red", linestyle='--')
    # ax.set_xlim([0, size.value])
    plt.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{output_directory}/{run_name}_contamination_hist{out_to_radius[0]}{out_to_radius[1]}.png")
    print(f"Saved: {output_directory}/{run_name}_contamination_hist{out_to_radius[0]}{out_to_radius[1]}.png")
    plt.close(fig)
    plt.close('all')


if __name__ == "__main__":
    
    run_name = 'L0300N0564_VR93'
    velociraptor_properties_zoom = "/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo/L0300N0564_VR93/properties"
    snap_filepath_zoom = "/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo/L0300N0564_VR93/snapshots/L0300N0564_VR93_0199.hdf5"
    output_directory = "/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"
    out_to_radius = (5, 'R200c')
    highres_radius = (6, 'R500c')

    contamination_map(
        run_name,
        velociraptor_properties_zoom,
        snap_filepath_zoom,
        out_to_radius=out_to_radius,
        highres_radius=highres_radius,
        output_directory=output_directory,
    )

    contamination_radial_histogram(
        run_name,
        velociraptor_properties_zoom,
        snap_filepath_zoom,
        out_to_radius=out_to_radius,
        highres_radius=highres_radius,
        output_directory=output_directory,
    )
