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

from convergence_radius import convergence_radius
from vr_rendezvous import find_object

# Constants
bins = 40
radius_bounds = [5e-3, 3]  # In units of R200crit
residual_bounds = [-0.5, 0.5]  # y-limits in the residual plot
cmap_name = 'BuPu_r'
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


def cumulative_mass_compare_plot(
        run_name: str,
        snap_filepath_parent: str = None,
        velociraptor_properties_parent: str = None,
        snap_filepath_zoom: List[str] = None,
        velociraptor_properties_zoom: List[str] = None,
        output_directory: str = None
) -> None:
    """
    This function compares the cumulative mass of DMO zooms to their
    corresponding parent halo. It also allows to assess numerical
    convergence by overlapping multiple profiles from zooms with different
    resolutions. The cumulative mass profiles are then listed in the legend,
    where the DM particle mass is quoted.
    The zoom inputs are in the form of arrays of strings, to allow
    for multiple entries. Each entry is a zoom snap/VR output resolution.
    The function allows not to plot either the parent mass profile
    or the zooms. At least one of them must be entered.

    :param run_name: str
        A custom and identifiable name for the run. Currently the standard
        name follows the scheme {AUTHOR_INITIALS}{HALO_ID}, e.g. SK0 or EA1.
        This argument must be defined.
    :param snap_filepath_parent: str
        The complete path to the snapshot of the parent box.
        If parameter is None, the mass profile of the parent box is not
        displayed.
    :param snap_filepath_zoom: list(str)
        The list of complete paths to the snapshots of the zooms
        at different resolution. Note: the order must match that of
        the `velociraptor_properties_zoom` parameter. If parameter
        is None, the mass profile of the zoom is not displayed and the
        `velociraptor_properties_zoom` parameter is ignored.
    :param velociraptor_properties_zoom: list(str)
        The list of complete paths to the VR outputs (properties) of the
        zooms at different resolution. Note: the order must match that of
        the `snap_filepath_zoom` parameter. If `snap_filepath_zoom` is None,
        then this parameter is ignored. If this parameter is None and
        `snap_filepath_zoom` is defined, raises an error.
    :param output_directory: str
        The output directory where to save the plot. This code assumes
        that the output directory exists. If it does not exist, matplotlib
        will return an error. This argument must be defined.
    :return: None
    """

    # ARGS CHECK #
    assert snap_filepath_parent or snap_filepath_zoom
    if snap_filepath_zoom and velociraptor_properties_zoom:
        assert len(snap_filepath_zoom) == len(velociraptor_properties_zoom)
    elif not snap_filepath_zoom:
        velociraptor_properties_zoom = None
    elif snap_filepath_zoom and not velociraptor_properties_zoom:
        raise ValueError
    assert output_directory

    fig, (ax, ax_residual) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7, 8),
        dpi=resolution // 8,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # PARENT #
    if snap_filepath_parent:
        # Rendezvous over parent VR catalogue using zoom information
        with h5py.File(velociraptor_properties_zoom[0], 'r') as vr_file:
            idx, M200c, R200c, Xcminpot, Ycminpot, Zcminpot = find_object(
                sample_structType=10,
                vr_properties_catalog=velociraptor_properties_parent,
                sample_mass_lower_lim=vr_file['/Mass_200crit'][0] * 1e10 * 0.8,
                sample_x=vr_file['/Xcminpot'][0],
                sample_y=vr_file['/Ycminpot'][0],
                sample_z=vr_file['/Zcminpot'][0],
            )

            M200c = unyt.unyt_quantity(M200c, unyt.Solar_Mass)
            R200c = unyt.unyt_quantity(R200c, unyt.Mpc)
            xCen = unyt.unyt_quantity(Xcminpot, unyt.Mpc)
            yCen = unyt.unyt_quantity(Ycminpot, unyt.Mpc)
            zCen = unyt.unyt_quantity(Zcminpot, unyt.Mpc)

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

        r = np.sqrt(wrap(posDM[:, 0] - xCen, data.metadata.boxsize[0]) ** 2 +
                    wrap(posDM[:, 1] - yCen, data.metadata.boxsize[1]) ** 2 +
                    wrap(posDM[:, 2] - zCen, data.metadata.boxsize[2]) ** 2)

        # Calculate particle mass and rho_crit
        unitLength = data.metadata.units.length
        unitMass = data.metadata.units.mass
        rho_crit = unyt.unyt_quantity(
            data.metadata.cosmology_raw['Critical density [internal units]'],
            unitMass / unitLength ** 3
        ).to('Msun/Mpc**3')
        rhoMean = rho_crit * data.metadata.cosmology.Om0
        vol = data.metadata.boxsize[0] ** 3
        numPart = data.metadata.n_dark_matter
        particleMass = rhoMean * vol / numPart
        parent_mass_resolution = particleMass
        particleMasses = np.ones_like(r.value) * particleMass

        # Construct bins and compute density profile
        lbins = np.logspace(np.log10(radius_bounds[0]), np.log10(radius_bounds[1]), bins)
        r_scaled = r / R200c
        hist, bin_edges = np.histogram(r_scaled, bins=lbins, weights=particleMasses)
        bin_centre = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        cumulative_mass_parent = np.cumsum(hist)

        # Plot density profile for each selected halo in volume
        cumulative_mass_parent[cumulative_mass_parent == 0] = np.nan
        parent_label = (
            f'Parent: $m_\\mathrm{{DM}} = {latex_float(parent_mass_resolution.value[0])}\\ '
            f'{parent_mass_resolution.units.latex_repr}$'
        )
        ax.plot(bin_centre, cumulative_mass_parent, c="grey", linestyle="-", label=parent_label)

        # Compute convergence radius
        conv_radius = convergence_radius(r, particleMasses, rho_crit) / R200c
        ax.axvline(conv_radius, color='grey', linestyle='--')
        ax_residual.axvline(conv_radius, color='grey', linestyle='--')
        t = ax.text(
            conv_radius,
            ax.get_ylim()[1],
            'Convergence radius',
            ha='center',
            va='top',
            rotation='vertical',
            alpha=0.6
        )
        t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # ZOOMS #
    if snap_filepath_zoom:

        # Set-up colors
        cmap_discrete = plt.cm.get_cmap(cmap_name, len(velociraptor_properties_zoom) + 3)
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

            r = np.sqrt(wrap(posDM[:, 0] - xCen, data.metadata.boxsize[0]) ** 2 +
                        wrap(posDM[:, 1] - yCen, data.metadata.boxsize[1]) ** 2 +
                        wrap(posDM[:, 2] - zCen, data.metadata.boxsize[2]) ** 2)

            # Calculate particle mass and rho_crit
            unitLength = data.metadata.units.length
            unitMass = data.metadata.units.mass
            rho_crit = unyt.unyt_quantity(
                data.metadata.cosmology_raw['Critical density [internal units]'],
                unitMass / unitLength ** 3
            ).to('Msun/Mpc**3')
            particleMasses = data.dark_matter.masses.to('Msun')
            zoom_mass_resolution = particleMasses

            # Construct bins and compute density profile
            lbins = np.logspace(np.log10(radius_bounds[0]), np.log10(radius_bounds[1]), bins)
            r_scaled = r / R200c
            hist, bin_edges = np.histogram(r_scaled, bins=lbins, weights=particleMasses)
            bin_centre = np.sqrt(bin_edges[1:] * bin_edges[:-1])
            cumulative_mass_zoom = np.cumsum(hist)

            # Plot mass profile for each selected halo in volume
            cumulative_mass_zoom[cumulative_mass_zoom == 0] = np.nan
            zoom_label = (
                f'Zoom: $m_\\mathrm{{DM}} = {latex_float(zoom_mass_resolution.value[0])}\\ '
                f'{zoom_mass_resolution.units.latex_repr}$'
            )
            ax.plot(bin_centre, cumulative_mass_zoom, c=color, linestyle="-", label=zoom_label)

            # Compute convergence radius
            conv_radius = convergence_radius(r, particleMasses, rho_crit) / R200c
            ax.axvline(conv_radius, color=color, linestyle='--')
            t = ax.text(
                conv_radius,
                ax.get_ylim()[1],
                'Convergence radius',
                ha='center',
                va='top',
                rotation='vertical',
                alpha=0.5
            )
            t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none'))

            # RESIDUALS #
            if snap_filepath_parent and snap_filepath_zoom:
                residual = (cumulative_mass_zoom - cumulative_mass_parent) / cumulative_mass_parent
                ax_residual.axhline(0, color='grey', linestyle='-')
                ax_residual.plot(bin_centre, residual, c=color, linestyle="-")
                ax_residual.axvline(conv_radius, color=color, linestyle='--')

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
        backgroundcolor='white',
        alpha=0.5,
        transform=ax.transAxes,
    )

    ax.axvline(1, color="grey", linestyle='--')
    ax_residual.axvline(1, color="grey", linestyle='--')
    ax_residual.set_xlim(radius_bounds[0], radius_bounds[1])
    ax_residual.set_ylim(residual_bounds[0], residual_bounds[1])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(f"$M_{{\\rm DM}} (< R)\\ [{M200c.units.latex_repr}]$")
    ax_residual.set_ylabel(f"$\\Delta M\\ /\\ M_{{\\rm parent}}\\ (< R)$")
    ax_residual.set_xlabel(r"$R\ /\ R_{200c}$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{output_directory}/{run_name}_cumulative_mass_compare.png")
    plt.close(fig)
    print(f"Saved: {output_directory}/{run_name}_cumulative_mass_compare.png")
    plt.close('all')

    return


if __name__ == "__main__":
    import sys

    # Argval inputs #
    # NOTE: The `snap_filepath_zoom` and `velociraptor_properties_zoom` options should be given
    # as long strings formed by the complete file-paths separated by a single comma and no spaces.
    # eg: path1,path2,path3 PATH1,PATH2,PATH3

    # run_name = sys.argv[1]
    # snap_filepath_parent = None
    # if sys.argv[2] != 'none':
    #     snap_filepath_parent = sys.argv[2]
    # snap_filepath_zoom = None
    # if sys.argv[3] != 'none':
    #     snap_filepath_zoom = sys.argv[3].split(',')
    # velociraptor_properties_zoom = None
    # if sys.argv[4] != 'none':
    #     velociraptor_properties_zoom = sys.argv[4].split(',')
    # output_directory = sys.argv[5]

    # Manual inputs #
    # NOTE: in loops, use one iteration per halo, i.e. one plot
    # produced per iteration. Gather snaps of the same cluster
    # at different resolutions in the same arrays, as they are
    # overplotted in the same figure.

    run_name = "L0300N0564_VR93"
    snap_filepath_parent = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/EAGLE-XL_L0300N0564_DMONLY_0036.hdf5"
    velociraptor_properties_parent = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/stf_swiftdm_3dfof_subhalo_0036/stf_swiftdm_3dfof_subhalo_0036.VELOCIraptor.properties.0"
    snap_filepath_zoom = [
        "/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo/L0300N0564_VR93/snapshots/L0300N0564_VR93_0199.hdf5"
    ]
    velociraptor_properties_zoom = [
        "/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo/L0300N0564_VR93/properties"
    ]
    output_directory = "/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"

    cumulative_mass_compare_plot(
        run_name,
        snap_filepath_parent=snap_filepath_parent,
        velociraptor_properties_parent=velociraptor_properties_parent,
        snap_filepath_zoom=snap_filepath_zoom,
        velociraptor_properties_zoom=velociraptor_properties_zoom,
        output_directory=output_directory
    )
