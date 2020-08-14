import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

try:
    plt.style.use("mnras.mplstyle")
except:
    pass

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from zooms import density_profiles as zoom
from parent import density_profiles as parent

for i in range(3):
    fig, ax = parent.density_profile(i, outfig=True)
    zoom.density_profile(i, outfig=True)
    fig.tight_layout()
    fig.savefig(f"{output_directory}halo{halo_id}{author}_density_profile_zoom.png")


