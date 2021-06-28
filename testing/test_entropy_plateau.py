import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("..")

from scaling_relations import EntropyPlateau
from register import find_files, set_mnras_stylesheet, xlargs


def set_snap_number(snap: str, cat: str, snap_number: int):
    old_snap_number = f"_{xlargs.snapshot_number:04d}"
    new_snap_number = f"_{snap_number:04d}"
    return snap.replace(old_snap_number, new_snap_number), cat.replace(old_snap_number, new_snap_number)


snap, cat = find_files()
set_mnras_stylesheet()

# Start from redshift 0.5 to select particles in the plateau
snap, cat = set_snap_number(snap, cat, 1482)
plateau = EntropyPlateau()
plateau.setup_data(path_to_snap=snap, path_to_catalogue=cat)
plateau.select_particles_on_plateau(shell_radius_r500=0.1, shell_thickness_r500=0.02, temperature_cut=True)
particle_ids_z0p5 = plateau.get_particle_ids()
agn_flag_z0p5 = plateau.get_heated_by_agnfeedback()
snii_flag_z0p5 = plateau.get_heated_by_sniifeedback()
print(f"Redshift {plateau.z:.3f}: {plateau.number_particles:d} particles selected")

num_snaps = 50
redshifts = np.empty(num_snaps)
snaps_collection = np.linspace(500, 1482, num_snaps, dtype=np.int)
particle_ids = np.empty((num_snaps, plateau.number_particles), dtype=np.int)
temperatures = np.empty((num_snaps, plateau.number_particles))
entropies = np.empty((num_snaps, plateau.number_particles))
hydrogen_number_densities = np.empty((num_snaps, plateau.number_particles))

del plateau

for i, new_snap_number in enumerate(snaps_collection[::-1]):
    # Move to high redshift and track the same particle IDs
    snap, cat = set_snap_number(snap, cat, new_snap_number)
    plateau = EntropyPlateau()
    plateau.setup_data(path_to_snap=snap, path_to_catalogue=cat)
    plateau.select_particles_on_plateau(particle_ids=particle_ids_z0p5, only_particle_ids=True)
    print(i, new_snap_number, f"Redshift {plateau.z:.3f}: {plateau.number_particles:d} particles selected")
    plateau.shell_properties()
    # plateau.heating_fractions(nbins=70)

    # Sort particles by ID
    sort_id = np.argsort(plateau.get_particle_ids())

    # Allocate data into arrays
    redshifts[i] = plateau.z
    particle_ids[i, :plateau.number_particles] = plateau.get_particle_ids()[sort_id]
    temperatures[i, :plateau.number_particles] = plateau.get_temperatures()[sort_id]
    entropies[i, :plateau.number_particles] = plateau.get_entropies()[sort_id]
    hydrogen_number_densities[i, :plateau.number_particles] = plateau.get_hydrogen_number_density()[sort_id]

    del plateau

hydrogen_number_densities_max = np.amax(hydrogen_number_densities, axis=0)

fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()
# plateau.plot_densities(axes)

bins = np.logspace(-4, 4, 64)
plt.yscale('log')
plt.xscale('log')
plt.hist(hydrogen_number_densities_max, bins=bins, label='All')
plt.hist(hydrogen_number_densities_max[snii_flag_z0p5], bins=bins, label='SN heated')
plt.hist(hydrogen_number_densities_max[agn_flag_z0p5], bins=bins, label='AGN heated')
plt.xlabel(r'$\log(n_{\rm H,max}/{\rm cm}^{-3})$')
axes.text(
    0.025,
    0.975,
    (
        f'z = {redshifts.max():.2f} - {redshifts.min():.2f}\n'
    ),
    color="k",
    ha="left",
    va="top",
    alpha=0.8,
    transform=axes.transAxes,
)
axes.legend(loc="upper right")

if not xlargs.quiet:
    plt.show()

plt.close()
