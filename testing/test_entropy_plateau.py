import sys
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
plateau = EntropyPlateau()
snap, cat = set_snap_number(snap, cat, 1482)
plateau.setup_data(path_to_snap=snap, path_to_catalogue=cat)
plateau.select_particles_on_plateau(shell_radius_r500=0.1, shell_thickness_r500=0.02, temperature_cut=True)
particle_ids_z0p5 = plateau.get_particle_ids()
print(f"Redshift {plateau.z:.3f}: {plateau.number_particles:d} particles selected")
print('number of ids', len(particle_ids_z0p5))
del plateau

# Move to redshift 3 and track the same particle IDs
plateau = EntropyPlateau()
snap, cat = set_snap_number(snap, cat, 987)
plateau.setup_data(path_to_snap=snap, path_to_catalogue=cat)
plateau.select_particles_on_plateau(particle_ids=particle_ids_z0p5, only_particle_ids=True)
print(f"Redshift {plateau.z:.3f}: {plateau.number_particles:d} particles selected")
plateau.shell_properties()
plateau.heating_fractions(nbins=50)

fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()
plateau.plot_densities(axes)

axes.text(
    0.025,
    0.975,
    (
        f'z = {plateau.z:.2f}\n'
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