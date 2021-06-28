import sys
from matplotlib import pyplot as plt

sys.path.append("..")

from scaling_relations import EntropyPlateau
from register import find_files, set_mnras_stylesheet, xlargs

snap, cat = find_files()
kwargs = dict(path_to_snap=snap, path_to_catalogue=cat)
set_mnras_stylesheet()

plateau = EntropyPlateau()
plateau.setup_data(**kwargs)
plateau.select_particles_on_plateau(shell_thickness_r500=0.05)
plateau.shell_properties()
plateau.heating_fractions()

fig = plt.figure(constrained_layout=True)
axes = fig.add_subplot()
plateau.plot_observations(axes)

if not xlargs.quiet:
    plt.show()

plt.close()