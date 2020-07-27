import numpy as np
import unyt
import swiftsimio as sw
from velociraptor.swift.swift import to_swiftsimio_dataset
from velociraptor.particles import load_groups
from velociraptor import load
from matplotlib import pyplot as plt

try:
    plt.style.use("mnras.mplstyle")
except:
    pass

print("Loading halos selected...")
lines = np.loadtxt("outfiles/halo_selected.txt", comments="#", delimiter=",", unpack=False).T
print("log10(M200c / Msun): ", np.log10(lines[1] * 1e13))
print("R200c: ", lines[2])
print("Centre of potential coordinates: (xC, yC, zC)")
for i in range(3):
    print(f"\tHalo {i:d}:\t({lines[3, i]:2.1f}, {lines[4, i]:2.1f}, {lines[5, i]:2.1f})")
M200c = lines[1] * 1e13
R200c = lines[2]
x = lines[3]
y = lines[4]
z = lines[5]

############################################################
# DENSITY PROFILE FROM SNAPSHOT - ALL PARTICLES
dataPath = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
snapFile = dataPath + "EAGLE-XL_L0300N0564_DMONLY_0036.hdf5"

for i in range(3):
    print(f"Profiling halo {i}...")
    xCen = unyt.unyt_quantity(x[i], unyt.Mpc)
    yCen = unyt.unyt_quantity(y[i], unyt.Mpc)
    zCen = unyt.unyt_quantity(z[i], unyt.Mpc)
    size = unyt.unyt_quantity(12. * R200c[i], unyt.Mpc)
    mask = sw.mask(snapFile)
    region = [
        [xCen - size, xCen + size],
        [yCen - size, yCen + size],
        [zCen - size, zCen + size]
    ]
    mask.constrain_spatial(region)
    data = sw.load(snapFile, mask=mask)
    posDM = data.dark_matter.coordinates# / data.metadata.a
    r = np.sqrt(posDM[:, 0] ** 2 + posDM[:, 1] ** 2 + posDM[:, 2] ** 2) / R200c[i]
    masses = np.ones_like(r)

    # constuct bins for the histogram
    lbins = np.logspace(-3, 1, 5)
    # compute statistics - each bin has Y value of the sum of the masses of points within the bin X
    hist, bin_edges = np.histogram(r, bins=lbins)
    print(lbins, hist)
    bin_centre = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    # compute radial density distribution
    volume_shell = (4. * np.pi * (R200c[i] ** 3) / 3.) * ((bin_edges[1:]) ** 3 - (bin_edges[:-1]) ** 3)
    rho_crit = data.metadata.cosmology['Critical density [internal units]'][0]
    densities = hist / volume_shell / rho_crit
    # Plot density profile for each selected halo in volume
    fig, ax = plt.subplots()
    ax.scatter(bin_centre, densities, color="C0", marker=".")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r"$\rho_{DM}/\rho_c$")
    ax.set_xlabel(r"$R/R_{200c}$")
    fig.tight_layout()
    fig.savefig(f"outfiles/halo{i}_density_profile_allparts.png")
    plt.close(fig)

############################################################
# DENSITY PROFILE FROM SNAPSHOT - SUBHALOS PARTICLES

# snapshot_path = "/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
# vrPath = snapshot_path + "stf_swiftdm_3dfof_subhalo_0036/"
# velociraptor_base_name = vrPath + "stf_swiftdm_3dfof_subhalo_0036.VELOCIraptor.properties.0"
# velociraptor_properties = velociraptor_base_name
# velociraptor_groups = velociraptor_base_name.replace("properties", "catalog_groups")
#
# filenames = {
#     "parttypes_filename": velociraptor_base_name.replace(
#         "properties", "catalog_partypes"
#     ),
#     "particles_filename": velociraptor_base_name.replace(
#         "properties", "catalog_particles"
#     ),
#     "unbound_parttypes_filename": velociraptor_base_name.replace(
#         "properties", "catalog_partypes.unbound"
#     ),
#     "unbound_particles_filename": velociraptor_base_name.replace(
#         "properties", "catalog_particles.unbound"
#     ),
# }
#
# catalogue = load(velociraptor_properties)
# groups = load_groups(velociraptor_groups, catalogue)
#
# for halo_id in halo_ids:
#     particles, unbound_particles = groups.extract_halo(halo_id, filenames=filenames)
#
#     M200c = catalogue.masses.mass_200mean[halo_id].to("Solar_Mass")
#     R200c = catalogue.apertures.mass_star_30_kpc[halo_id].to("Solar_Mass")
#
#     # This reads particles using the cell metadata that are around our halo
#     data = to_swiftsimio_dataset(particles, snapshot_path, generate_extra_mask=False)
#     x = particles.x / data.metadata.a
#     y = particles.y / data.metadata.a
#     z = particles.z / data.metadata.a
#     r = np.sqrt(x ** 2 + y ** 2 + z ** 2) / R200c
#     masses = np.ones_like(r)
