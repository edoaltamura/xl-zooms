import matplotlib
matplotlib.use('Agg')

import unyt
import h5py

import swiftsimio as sw
from swiftsimio.visualisation.projection import project_pixel_grid
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

#############################################
# INPUTS
author = "SK"
out_to_radius = 5
run_name = "EAGLE-XL_ClusterSK0_+1res"
metadata_filepath = f"outfiles/halo_selected_{author}.txt"
simdata_dirpath = "/cosma7/data/dp004/rttw52/swift_runs/runs/EAGLE-XL/"
snap_relative_filepaths = ['EAGLE-XL_ClusterSK0_High/snapshots/EAGLE-XL_ClusterSK0_High_0001.hdf5']
velociraptor_properties = ["/cosma6/data/dp004/dc-alta2/xl-zooms/halo_SK0_High_0001/halo_SK0_High_0001.properties.0"]
output_directory = "/cosma7/data/dp004/dc-alta2/xl-zooms/analysis"

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
    gas_map = project_pixel_grid(
        # Note here that we pass in the dark matter dataset not the whole
        # data object, to specify what particle type we wish to visualise
        data=data.gas,
        boxsize=data.metadata.boxsize,
        resolution=data,
        project=None,
        parallel=True,
        region=(region[0] + region[1])
    )

    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=1024 // 8)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    ax.imshow(gas_map.T, norm=LogNorm(), cmap="inferno", origin="lower", extent=(region[0] + region[1]))
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
        transform=ax.transAxes,
    )
    ax.text(
        xCen.value,
        yCen.value + 1.05 * R200c[i],
        r"$R_{200c}$",
        color="white",
        ha="center",
        va="bottom"
    )
    ax.text(
        xCen.value,
        yCen.value + 1.05 * 5 * R200c[i],
        r"$5 \times R_{200c}$",
        color="grey",
        ha="center",
        va="bottom"
    )
    circle_r200 = plt.Circle((xCen, yCen), R200c[i], color="white", fill=False, linestyle='-')
    circle_5r200 = plt.Circle((xCen, yCen), 5 * R200c[i], color="grey", fill=False, linestyle='--')
    ax.add_artist(circle_r200)
    ax.add_artist(circle_5r200)
    fig.savefig(f"{output_directory}/halo{author}{i}_DMmap{out_to_radius}r200_zoom.png")
    plt.close(fig)
