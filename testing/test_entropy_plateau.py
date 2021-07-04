import sys
import os, psutil
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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
_snap, _cat = set_snap_number(snap, cat, 1482)
plateau = EntropyPlateau()
plateau.setup_data(path_to_snap=_snap, path_to_catalogue=_cat)
plateau.select_particles_on_plateau(shell_radius_r500=0.1, shell_thickness_r500=0.02, temperature_cut=True)
particle_ids_z0p5 = plateau.get_particle_ids()
agn_flag_z0p5 = plateau.get_heated_by_agnfeedback()
snii_flag_z0p5 = plateau.get_heated_by_sniifeedback()
number_particles_z0p5 = plateau.number_particles
print(f"Redshift {plateau.z:.3f}: {plateau.number_particles:d} particles selected")

snaps_collection = np.arange(400, 1842, 200)
num_snaps = len(snaps_collection)
redshifts = np.empty(num_snaps)
particle_ids = np.empty((num_snaps, plateau.number_particles), dtype=np.int)
temperatures = np.empty((num_snaps, plateau.number_particles))
entropies = np.empty((num_snaps, plateau.number_particles))
hydrogen_number_densities = np.empty((num_snaps, plateau.number_particles))

del plateau

for i, new_snap_number in enumerate(snaps_collection[::-1]):
    # Move to high redshift and track the same particle IDs
    _snap, _cat = set_snap_number(snap, cat, new_snap_number)
    plateau = EntropyPlateau()
    plateau.setup_data(path_to_snap=_snap, path_to_catalogue=_cat)
    plateau.select_particles_on_plateau(particle_ids=particle_ids_z0p5, only_particle_ids=True)
    print(i, new_snap_number, f"Redshift {plateau.z:.3f}: {plateau.number_particles:d} particles selected")
    plateau.shell_properties()

    # Sort particles by ID
    sort_id = np.argsort(plateau.get_particle_ids())

    # Allocate data into arrays
    redshifts[i] = plateau.z
    # particle_ids[i, :plateau.number_particles] = plateau.get_particle_ids()[sort_id]
    temperatures[i, :plateau.number_particles] = plateau.get_temperatures()[sort_id]
    entropies[i, :plateau.number_particles] = plateau.get_entropies()[sort_id]
    hydrogen_number_densities[i, :plateau.number_particles] = plateau.get_hydrogen_number_density()[sort_id]

    process = psutil.Process(os.getpid())
    print(f"Memory used: {process.memory_info().rss / 1024 / 1024:.3f} MB")

    del plateau

hydrogen_number_densities_max = np.amax(hydrogen_number_densities, axis=0)

fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()
bins = np.logspace(-4, 4, 64)
axes.set_yscale('log')
axes.set_xscale('log')
axes.hist(
    hydrogen_number_densities_max, bins=bins,
    label=f'All ({number_particles_z0p5:d} particles)'
)
axes.hist(
    hydrogen_number_densities_max[snii_flag_z0p5], bins=bins,
    label=f'SNe heated ({snii_flag_z0p5.sum() / number_particles_z0p5 * 100:.1f} %)'
)
axes.hist(
    hydrogen_number_densities_max[agn_flag_z0p5], bins=bins,
    label=f'AGN heated ({agn_flag_z0p5.sum() / number_particles_z0p5 * 100:.1f} %)'
)
axes.set_xlabel(r'$n_{\rm H,max}$ [cm$^{-3}$]')
axes.set_ylabel(f"Number of particles")
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

# if not xlargs.quiet:
#     plt.show()

plt.savefig('max_density_track.pdf')

plt.close()

fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()
axes.set_yscale('log')
axes.set_xlabel('Redshift')
axes.set_ylabel(f"Temperature")
axes.set_ylim(1e3, 1e10)
axes.plot(temperatures.T[snii_flag_z0p5], color='g', linewidth=0.1, alpha=0.2)
axes.plot(temperatures.T[agn_flag_z0p5], color='r', linewidth=0.1, alpha=0.2)

if not xlargs.quiet:
    plt.show()

plt.savefig('temperature_track.pdf')

plt.close()