import matplotlib

matplotlib.use('Agg')

import numpy as np
import unyt
import h5py
import matplotlib.pyplot as plt
import swiftsimio as sw


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


def map_setup_axes(ax: plt.Axes, halo_id: int, redshift: float, M200c: float, R200c: float) -> None:
    ax.set_aspect('equal')
    ax.set_ylabel(r"$y$ [Mpc]")
    ax.set_xlabel(r"$x$ [Mpc]")
    ax.text(
        0.025,
        0.975,
        f"Halo {halo_id:d} DMO\n",
        color="black",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.975,
        0.975,
        f"$z={redshift:3.3f}$",
        color="black",
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.975,
        0.025,
        (
            f"$M_{{200c}}={latex_float(M200c)}$ M$_\odot$"
        ),
        color="black",
        ha="right",
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
    circle_r200 = plt.Circle((0, 0), R200c, color="black", fill=False, linestyle='-')
    circle_5r200 = plt.Circle((0, 0), 5 * R200c, color="grey", fill=False, linestyle='--')
    ax.add_artist(circle_r200)
    ax.add_artist(circle_5r200)

    return

def hist_setup_axes(ax: plt.Axes, halo_id: int, redshift: float, M200c: float, R200c: float) -> None:
    ax.set_yscale('log')
    ax.set_ylabel("Number of particles")
    ax.set_xlabel(r"$R\ /\ R_{200c}$")
    ax.axvline(1, color="grey", linestyle='--')

    ax.text(
        0.025,
        0.025,
        (
            f"Halo {halo_id:d} DMO\n"
            f"$z={redshift:3.3f}$\n"
            f"$M_{{200c}}={latex_float(M200c)}$ M$_\odot$\n"
            f"$R_{{200c}}={latex_float(R200c)}$ Mpc"
        ),
        color="black",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )
    return


#############################################
# INPUTS
author = "SK"
highres_radius = 3 # in Mpc
out_to_radius = 7 # in R200crit units
boxMpc = 300. # in Mpc

metadata_filepath = f"outfiles/halo_selected_{author}.txt"
simdata_dirpath = "/cosma6/data/dp004/rttw52/EAGLE-XL/"

snap_relative_filepaths = [
    f"EAGLE-XL_Cluster{author}{i}_DMO/snapshots/EAGLE-XL_Cluster{author}{i}_DMO_0001.hdf5"
    for i in range(3)
]

velociraptor_properties = [
    f"/cosma6/data/dp004/dc-alta2/xl-zooms/halo_{author}{i}_0001/halo_{author}{i}_0001.properties.0"
    for i in range(3)
]

output_directory = "outfiles/"

#############################################

print("Loading halos selected...")
M200c = []
R200c = []
x = []
y = []
z = []

for vr_path in velociraptor_properties:
    with h5py.File(vr_path, 'r') as vr_file:
        M200c.append(vr_file['/Mass_200crit'][0] * 1e10)
        R200c.append(vr_file['/R_200crit'][0])
        x.append(vr_file['/Xcminpot'][0])
        y.append(vr_file['/Ycminpot'][0])
        z.append(vr_file['/Zcminpot'][0])

for i in range(len(snap_relative_filepaths)):
    # EAGLE-XL data path
    snapFile = simdata_dirpath + snap_relative_filepaths[i]
    print(f"Rendering {snap_relative_filepaths[i]}...")
    # Load data using mask
    xCen = unyt.unyt_quantity(x[i], unyt.Mpc)
    yCen = unyt.unyt_quantity(y[i], unyt.Mpc)
    zCen = unyt.unyt_quantity(z[i], unyt.Mpc)
    size = unyt.unyt_quantity(out_to_radius * R200c[i], unyt.Mpc)
    mask = sw.mask(snapFile)
    region = [
        [xCen - size, xCen + size],
        [yCen - size, yCen + size],
        [zCen - size, zCen + size]
    ]
    mask.constrain_spatial(region)

    # Load data using mask
    data = sw.load(snapFile, mask=mask)
    posDM = data.dark_matter.coordinates / data.metadata.a
    highres_coordinates = {
        'x': wrap(posDM[:, 0] - xCen, boxMpc),
        'y': wrap(posDM[:, 1] - yCen, boxMpc),
        'z': wrap(posDM[:, 2] - zCen, boxMpc),
        'r': np.sqrt(wrap(posDM[:, 0] - xCen, boxMpc) ** 2 +
                     wrap(posDM[:, 1] - yCen, boxMpc) ** 2 +
                     wrap(posDM[:, 2] - zCen, boxMpc) ** 2)
    }
    del posDM
    posDM = data.boundary.coordinates / data.metadata.a
    lowres_coordinates = {
        'x': wrap(posDM[:, 0] - xCen, boxMpc),
        'y': wrap(posDM[:, 1] - yCen, boxMpc),
        'z': wrap(posDM[:, 2] - zCen, boxMpc),
        'r': np.sqrt(wrap(posDM[:, 0] - xCen, boxMpc) ** 2 +
                     wrap(posDM[:, 1] - yCen, boxMpc) ** 2 +
                     wrap(posDM[:, 2] - zCen, boxMpc) ** 2)
    }
    del posDM

    # Flag contamination particles within 5 R200
    contaminated_idx = np.where(lowres_coordinates['r'] < highres_radius)[0]
    contaminated_r200_idx = np.where(lowres_coordinates['r'] < 1. * R200c[i])[0]
    print(f"Total low-res DM: {len(lowres_coordinates['r'])} particles detected")
    print(f"Contaminating low-res DM (< R_clean): {len(contaminated_idx)} particles detected")
    print(f"Contaminating low-res DM (< R200c): {len(contaminated_r200_idx)} particles detected")

    # Make particle maps
    fig, ax = plt.subplots(figsize=(8, 8), dpi=1024 // 8)
    map_setup_axes(ax, i, data.metadata.z, M200c[i], R200c[i])

    ax.plot(highres_coordinates['x'], highres_coordinates['y'], ',', c="C0", alpha=0.2, label='Highres')
    ax.plot(lowres_coordinates['x'][contaminated_idx], lowres_coordinates['y'][contaminated_idx], 'x', c="red", alpha=1, label='Lowres contaminating')
    ax.plot(lowres_coordinates['x'][~contaminated_idx], lowres_coordinates['y'][~contaminated_idx], '.', c="green", alpha=0.2, label='Lowres clean')

    ax.set_xlim([-size.value, size.value])
    ax.set_ylim([-size.value, size.value])
    plt.legend()
    fig.savefig(f"{output_directory}{author}{i}_contamination_map{out_to_radius}r200_zoom.png")
    plt.close(fig)

    # Histograms
    bins = np.linspace(0, out_to_radius * R200c[i], 40)
    hist, bin_edges = np.histogram(lowres_coordinates['r'][contaminated_idx], bins=bins)
    lowres_coordinates['r_bins'] = (bin_edges[1:] + bin_edges[:-1]) / 2 / R200c[i]
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
    fig, ax = plt.subplots()
    hist_setup_axes(ax, i, data.metadata.z, M200c[i], R200c[i])

    ax.step(highres_coordinates['r_bins'], highres_coordinates['hist_all'], where='mid', color='grey', label='Highres all')
    ax.step(lowres_coordinates['r_bins'], lowres_coordinates['hist_all'], where='mid', color='green', label='Lowres all')
    ax.step(lowres_coordinates['r_bins'], lowres_coordinates['hist_contaminating'], where='mid', color='red', label='Lowres contaminating')
    ax.axvline(highres_radius / R200c[i], color="grey", linestyle='--')
    ax.set_xlim([0, out_to_radius])
    plt.legend()
    fig.tight_layout()
    fig.savefig(f"{output_directory}{author}{i}_contamination_hist_{out_to_radius}r200_zoom.png")
    plt.close(fig)
