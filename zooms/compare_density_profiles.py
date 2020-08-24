# DENSITY PROFILE FROM SNAPSHOT - COMPARISON PARENT | ZOOM

import matplotlib

matplotlib.use('Agg')

from typing import List
import numpy as np
import unyt
import h5py
import swiftsimio as sw
from matplotlib import pyplot as plt

try:
    plt.style.use("mnras.mplstyle")
except:
    pass

# Constants
bins = 40
radius_bounds = [1e-2, 3]  # In units of R200crit
cmap_name = 'BuPu'


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def density_profile_compare_plot(
        run_name: str,
        snap_filepath_parent: str = None,
        snap_filepath_zoom: List[str] = None,
        velociraptor_properties_zoom: List[str] = None,
        output_directory: str = None
) -> None:
    """
    This function compares the density profiles of DMO zooms to their
    corresponding parent halo. It also allows to assess numerical
    convergence by overlapping multiple profiles from zooms with different
    resolutions. The density profiles are then listed in the legend,
    where the DM particle mass is quoted.
    The zoom inputs are in the form of arrays of strings, to allow
    for multiple entries. Each entry is a zoom snap/VR output resolution.

    :param run_name: str
        A custom and identifiable name for the run.
    :param snap_filepath_parent: str
        The complete path to the snapshot of the parent box.
    :param snap_filepath_zoom: list(str)
        The list of complete paths to the snapshots of the zooms
        at different resolution. Note: the order must match that of
        the `velociraptor_properties_zoom` parameter.
    :param velociraptor_properties_zoom: list(str)
        The list of complete paths to the VR outputs (properties) of the
        zooms at different resolution. Note: the order must match that of
        the `snap_filepath_zoom` parameter.
    :param output_directory: str
        The output directory where to save the plot.
    :return: None
    """

    # TEMPORARY
    # Split the run_name into author and halo_id to make everything work fine for now
    import re
    match = re.match(r"([a-z]+)([0-9]+)", run_name, re.I)
    author = None
    halo_id = None
    if match:
        author, halo_id = match.groups()


    # PARENT #
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
    ax.plot(bin_centre, densities, c="grey", linestyle="-", label=parent_label)


    # ZOOMS #

    # Set-up colors
    cmap_discrete = plt.cm.get_cmap(cmap_name, len(velociraptor_properties_zoom))
    cmaplist = [cmap_discrete(i) for i in range(cmap_discrete.N)]

    for snap_path, vrprop_path, color in zip(snap_filepath_zoom, velociraptor_properties_zoom, cmaplist):

        # Load velociraptor data
        with h5py.File(vrprop_path, 'r') as vr_file:
            M200c = vr_file['/Mass_200crit'][0] * 1e10
            R200c = vr_file['/R_200crit'][0]
            Xcminpot = vr_file['/Xcminpot'][0]
            Ycminpot = vr_file['/Ycminpot'][0]
            Zcminpot = vr_file['/Zcminpot'][0]

        M200c = unyt.unyt_quantity(M200c, unyt.Solar_Mass)
        R200c = unyt.unyt_quantity(R200c, unyt.Mpc)
        xCen = unyt.unyt_quantity(Xcminpot, unyt.Mpc)
        yCen = unyt.unyt_quantity(Ycminpot, unyt.Mpc)
        zCen = unyt.unyt_quantity(Zcminpot, unyt.Mpc)

        # Construct spatial mask to feed into swiftsimio
        size = radius_bounds[1] * R200c
        mask = sw.mask(snap_path)
        region = [
            [xCen - size, xCen + size],
            [yCen - size, yCen + size],
            [zCen - size, zCen + size]
        ]
        mask.constrain_spatial(region)
        data = sw.load(snap_path, mask=mask)

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
        particleMasses = data.dark_matter.masses.to('Msun')
        zoom_mass_resolution = particleMasses

        # Construct bins and compute density profile
        lbins = np.logspace(np.log10(radius_bounds[0]), np.log10(radius_bounds[1]), bins)
        hist, bin_edges = np.histogram(r, bins=lbins, weights=particleMasses)
        bin_centre = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        volume_shell = (4. * np.pi / 3.) * (R200c ** 3) * ((bin_edges[1:]) ** 3 - (bin_edges[:-1]) ** 3)
        densities = hist / volume_shell / rho_crit

        # Plot density profile for each selected halo in volume
        zoom_label = f'Zoom: $m_\\mathrm{{DM}} = {latex_float(zoom_mass_resolution.value[0])}\\ {zoom_mass_resolution.units.latex_repr}$'
        ax.plot(bin_centre, densities, c=color, linestyle="-", label=zoom_label)

    ax.text(
        0.025,
        0.025,
        (
            f"Halo {run_name:s} DMO\n"
            f"$z={data.metadata.z:3.3f}$\n"
            "Zoom VR output:\n"
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
    fig.savefig(f"{output_directory}/{run_name}_density_profile_compare.png")
    plt.close(fig)

    return


if __name__ == "__main__":
    # import sys

    # Argval inputs #
    # NOTE: The `snap_filepath_zoom` and `velociraptor_properties_zoom` options should be given
    # as long strings formed by the complete file-paths separated by a single comma and no spaces.
    # eg: path1,path2,path3 PATH1,PATH2,PATH3

    # run_name = sys.argv[1]
    # snap_filepath_parent = sys.argv[2]
    # snap_filepath_zoom = sys.argv[3].split(',')
    # velociraptor_properties_zoom = sys.argv[4].split(',')
    # output_directory = sys.argv[5]

    # Manual inputs #
    # NOTE: in loops, use one iteration per halo, i.e. one plot
    # produced per iteration. Gather snaps of the same cluster
    # at different resolutions in the same arrays, as they are
    # overplotted in the same figure.

    for i in range(3):

        halo_id = i
        run_name = f"SK{i}"
        snap_filepath_parent = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/EAGLE-XL_L0300N0564_DMONLY_0036.hdf5"
        snap_filepath_zoom = [f"/cosma6/data/dp004/rttw52/EAGLE-XL/EAGLE-XL_ClusterSK{i}_DMO/snapshots/EAGLE-XL_ClusterSK{i}_DMO_0001.hdf5"]
        velociraptor_properties_zoom = [f"/cosma6/data/dp004/dc-alta2/xl-zooms/halo_SK{i}_0001/halo_SK{i}_0001.properties.0"]
        output_directory = "outfiles"

        density_profile_compare_plot(
            run_name,
            snap_filepath_parent=snap_filepath_parent,
            snap_filepath_zoom=snap_filepath_zoom,
            velociraptor_properties_zoom=velociraptor_properties_zoom,
            output_directory=output_directory
        )
