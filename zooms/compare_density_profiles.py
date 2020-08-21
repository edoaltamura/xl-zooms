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
from parent import density_profiles as parentbox

author = "SK"
out_to_radius = 3
output_directory = "outfiles/"


for i in range(3):
    fig, ax = parentbox.density_profile(i, outfig=True)
    zoom.density_profile(i, outfig=True)
    fig.tight_layout()
    fig.savefig(f"{output_directory}halo{i}{author}_density_profile_compare_zoom.png")
    plt.close(fig)


