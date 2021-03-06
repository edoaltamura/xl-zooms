# DENSITY PROFILE FROM SNAPSHOT - Parent

import matplotlib

matplotlib.use('Agg')

import numpy as np
import unyt
import swiftsimio as sw
from matplotlib import pyplot as plt

try:
    plt.style.use("mnras.mplstyle")
except:
    pass


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


# Constants
bins = 40
radius_bounds = [1e-2, 3]  # In units of R200crit


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def density_profile_parent_plot(
        halo_id: int,
        author: str,
        snap_filepath_parent: str = None,
        output_directory: str = None
) -> None:

    # Load VR output gathered from the halo selection process
    lines = np.loadtxt(f"{output_directory}/halo_selected_{author}.txt", comments="#", delimiter=",", unpack=False).T
    M200c = lines[1] * 1e13
    R200c = lines[2]
    Xcminpot = lines[3]
    Ycminpot = lines[4]
    Zcminpot = lines[5]
    M200c = unyt.unyt_quantity(M200c[halo_id], unyt.Solar_Mass)
    R200c = unyt.unyt_quantity(R200c[halo_id], unyt.Mpc)
    xCen = unyt.unyt_quantity(Xcminpot[halo_id], unyt.Mpc)
    yCen = unyt.unyt_quantity(Ycminpot[halo_id], unyt.Mpc)
    zCen = unyt.unyt_quantity(Zcminpot[halo_id], unyt.Mpc)

    # Construct spatial mask to feed into swiftsimio
    size = radius_bounds[1] * R200c
    mask = sw.mask(snap_filepath_parent)
    region = [
        [xCen - size, xCen + size],
        [yCen - size, yCen + size],
        [zCen - size, zCen + size]
    ]
    mask.constrain_spatial(region)
    data = sw.load(snap_filepath_parent, mask=mask)

    # Get DM particle coordinates and compute radial distance from CoP in R200 units
    posDM = data.dark_matter.coordinates / data.metadata.a
    r = np.sqrt(
        (posDM[:, 0] - xCen) ** 2 +
        (posDM[:, 1] - yCen) ** 2 +
        (posDM[:, 2] - zCen) ** 2
    ) / R200c

    # Calculate particle mass and rho_crit
    unitLength = data.metadata.units.length
    unitMass = data.metadata.units.mass
    rho_crit = unyt.unyt_quantity(
        data.metadata.cosmology['Critical density [internal units]'],
        unitMass / unitLength ** 3
    )
    rhoMean = rho_crit * data.metadata.cosmology['Omega_m']
    vol = data.metadata.boxsize[0] ** 3
    numPart = data.metadata.n_dark_matter
    particleMass = rhoMean * vol / numPart
    parent_mass_resolution = particleMass

    # Construct bins and compute density profile
    lbins = np.logspace(np.log10(radius_bounds[0]), np.log10(radius_bounds[1]), bins)
    hist, bin_edges = np.histogram(r, bins=lbins)
    bin_centre = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    volume_shell = (4. * np.pi / 3.) * (R200c ** 3) * ((bin_edges[1:]) ** 3 - (bin_edges[:-1]) ** 3)
    densities = hist * particleMass / volume_shell / rho_crit

    # Plot density profile for each selected halo in volume
    fig, ax = plt.subplots()
    parent_label = f'Parent: $m_\\mathrm{{DM}} = {latex_float(parent_mass_resolution.value[0])}\\ {parent_mass_resolution.units.latex_repr}$'
    ax.plot(bin_centre, densities, c="lime", linestyle="-", label=parent_label)

    ax.text(
        0.025,
        0.025,
        (
            f"Halo {halo_id:d} DMO\n"
            f"$z={data.metadata.z:3.3f}$\n"
            "Parent VR output:\n"
            f"$M_{{200c}}={latex_float(M200c.value)}\\ {M200c.units.latex_repr}$\n"
            f"$R_{{200c}}={latex_float(R200c.value)}\\ {R200c.units.latex_repr}$"
        ),
        color="black",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.set_xlim(radius_bounds[0], radius_bounds[1])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r"$\rho_{DM}\ /\ \rho_c$")
    ax.set_xlabel(r"$R\ /\ R_{200c}$")
    plt.legend()
    fig.tight_layout()
    fig.savefig(f"{output_directory}/halo{halo_id}{author}_density_profile_parent.png")
    plt.close(fig)

    return


if __name__ == "__main__":
    # import sys

    # snap_filepath_parent = sys.argv[1]
    # snap_filepath_zoom = sys.argv[2]
    # velociraptor_properties_zoom = sys.argv[3]
    # output_directory = sys.argv[4]

    for i in range(3):
        # Manual inputs
        halo_id = i
        author = "SK"
        snap_filepath_parent = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/EAGLE-XL_L0300N0564_DMONLY_0036.hdf5"
        output_directory = "outfiles"

        density_profile_parent_plot(
            i,
            author,
            snap_filepath_parent=snap_filepath_parent,
            output_directory=output_directory
        )
