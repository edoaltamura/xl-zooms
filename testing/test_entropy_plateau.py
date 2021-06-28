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

plateau = EntropyPlateau()
print(snap, cat)
snap, cat = set_snap_number(snap, cat, 200)
print(snap, cat)
plateau.setup_data(path_to_snap=snap, path_to_catalogue=cat)
plateau.select_particles_on_plateau(shell_radius_r500=0.1, shell_thickness_r500=0.02, temperature_cut=True)
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