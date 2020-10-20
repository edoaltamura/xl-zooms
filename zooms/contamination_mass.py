import matplotlib

matplotlib.use('Agg')

import numpy as np
import unyt
import h5py
import matplotlib.pyplot as plt
import swiftsimio as sw

try:
    plt.style.use("mnras.mplstyle")
except:
    pass

#############################################
# INPUTS
author = "SK"
highres_radius = 4 # in Mpc
out_to_radius = 5 # in R200crit units
boxMpc = 300. # in Mpc



#############################################


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


def map_setup_axes(ax: plt.Axes, run_name: str, redshift: float, M200c: float, R200c: float) -> None:
    ax.set_aspect('equal')
    ax.set_ylabel(r"$y$ [Mpc]")
    ax.set_xlabel(r"$x$ [Mpc]")
    ax.text(
        0.025,
        0.025,
        (
            f"Halo {run_name:s} DMO\n"
            f"$z={redshift:3.3f}$\n"
            f"$M_{{200c}}={latex_float(M200c.value)}$ M$_\odot$\n"
            f"$R_{{200c}}={latex_float(R200c.value)}$ Mpc\n"
            f"$R_{{\\rm clean}}={latex_float(highres_radius)}$ Mpc"
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
        0 + 1.02 * highres_radius,
        r"$R_\mathrm{clean}$",
        color="red",
        ha="center",
        va="bottom"
    )
    circle_r200 = plt.Circle((0, 0), R200c, color="black", fill=False, linestyle='-')
    circle_5r200 = plt.Circle((0, 0), 5 * R200c, color="grey", fill=False, linestyle='--')
    circle_clean = plt.Circle((0, 0), highres_radius, color="red", fill=False, linestyle=':')
    ax.add_artist(circle_r200)
    ax.add_artist(circle_5r200)
    ax.add_artist(circle_clean)

    return

def hist_setup_axes(ax: plt.Axes, run_name: str, redshift: float, M200c: float, R200c: float) -> None:
    ax.set_yscale('log')
    ax.set_ylabel("Number of particles")
    ax.set_xlabel(r"$R\ /\ R_{200c}$")
    ax.axvline(1, color="grey", linestyle='--')
    ax.text(
        0.025,
        0.025,
        (
            f"Halo {run_name:s} DMO\n"
            f"$z={redshift:3.3f}$\n"
            f"$M_{{200c}}={latex_float(M200c.value)}$ M$_\odot$\n"
            f"$R_{{200c}}={latex_float(R200c.value)}$ Mpc\n"
            f"$R_{{\\rm clean}}={latex_float(highres_radius)}$ Mpc"
        ),
        color="black",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )
    return


velociraptor_properties_zoom = "/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo/L0300N0564_VR93/properties"
snap_filepath_zoom = "/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo/L0300N0564_VR93/snapshots/L0300N0564_VR93_0199.hdf5"
output_directory = "/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"
out_to_radius = 5
run_name = 'L0300N0564_VR93'

print("Loading halos selected...")
# Rendezvous over parent VR catalogue using zoom information
with h5py.File(velociraptor_properties_zoom, 'r') as vr_file:

    M200c = unyt.unyt_quantity(vr_file['/Mass_200crit'][0] * 1e10, unyt.Solar_Mass)
    R200c = unyt.unyt_quantity(vr_file['/R_200crit'][0], unyt.Mpc)
    xCen = unyt.unyt_quantity(vr_file['/Xcminpot'][0], unyt.Mpc)
    yCen = unyt.unyt_quantity(vr_file['/Ycminpot'][0], unyt.Mpc)
    zCen = unyt.unyt_quantity(vr_file['/Zcminpot'][0], unyt.Mpc)
    highres_radius = 6 * vr_file['/SO_R_500_rhocrit'][0]

# EAGLE-XL data path
print(f"Rendering {snap_filepath_zoom}...")
size = unyt.unyt_quantity(out_to_radius * R200c, unyt.Mpc)
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
contaminated_r200_idx = np.where(lowres_coordinates['r'] < 1. * R200c)[0]
print(f"Total low-res DM: {len(lowres_coordinates['r'])} particles detected")
print(f"Contaminating low-res DM (< R_clean): {len(contaminated_idx)} particles detected")
print(f"Contaminating low-res DM (< R200c): {len(contaminated_r200_idx)} particles detected")

# Make particle maps
fig, ax = plt.subplots(figsize=(7, 7), dpi=1024 // 7)
map_setup_axes(ax, run_name, data.metadata.z, M200c, R200c)

ax.plot(highres_coordinates['x'], highres_coordinates['y'], ',', c="C0", alpha=0.2, label='Highres')
ax.plot(lowres_coordinates['x'][contaminated_idx], lowres_coordinates['y'][contaminated_idx], 'x', c="red", alpha=1, label='Lowres contaminating')
ax.plot(lowres_coordinates['x'][~contaminated_idx], lowres_coordinates['y'][~contaminated_idx], '.', c="green", alpha=0.2, label='Lowres clean')

ax.set_xlim([-size.value, size.value])
ax.set_ylim([-size.value, size.value])
plt.legend()
fig.savefig(f"{output_directory}/{run_name}_contamination_map{out_to_radius}r200_zoom.png")
plt.close(fig)

# Histograms
bins = np.linspace(0, out_to_radius * R200c, 40)
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
fig, ax = plt.subplots()
hist_setup_axes(ax, run_name, data.metadata.z, M200c, R200c)

ax.step(highres_coordinates['r_bins'], highres_coordinates['hist_all'], where='mid', color='grey', label='Highres all')
ax.step(lowres_coordinates['r_bins'], lowres_coordinates['hist_all'], where='mid', color='green', label='Lowres all')
ax.step(lowres_coordinates['r_bins'], lowres_coordinates['hist_contaminating'], where='mid', color='red', label='Lowres contaminating')
ax.axvline(highres_radius / R200c, color="grey", linestyle='--')
ax.set_xlim([0, out_to_radius])
plt.legend()
fig.tight_layout()
fig.savefig(f"{output_directory}/{run_name}_contamination_hist_{out_to_radius}r200_zoom.png")
plt.close(fig)
