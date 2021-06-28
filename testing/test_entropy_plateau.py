import sys
from matplotlib import pyplot as plt

sys.path.append("..")

from scaling_relations import EntropyPlateau
from register import find_files, set_mnras_stylesheet, xlargs, Tcut_halogas

snap, cat = find_files()
kwargs = dict(path_to_snap=snap, path_to_catalogue=cat)
set_mnras_stylesheet()

plateau = EntropyPlateau()
plateau.setup_data(**kwargs)
plateau.select_particles_on_plateau(shell_radius_r500=0.1, shell_thickness_r500=0.02, temperature_cut=False)
plateau.shell_properties()
plateau.heating_fractions()

fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()
plateau.plot_densities(axes)

axes.text(
    0.025,
    0.975,
    (
        f'Shell thickness: {0.02:.3f} $r_{{500}}$\n'
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