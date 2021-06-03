import sys

sys.path.append("..")

from literature import *
from register import set_mnras_stylesheet



lit = Sun2009()
lit.filter_by('M_500', 8e13, 3e14)
# lit.overlay_points(None, 'T_2500', 'K_30kpc')
lit.overlay_entropy_profiles(k_units='K500adi')