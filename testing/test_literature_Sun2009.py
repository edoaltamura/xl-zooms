import sys

sys.path.append("..")

from literature import *

# Sun2009().overlay_entropy_profiles(k_units='K500adi')

lit = Sun2009()
# lit.filter_by('M_500', 8e13, 3e14)
lit.overlay_points(None, 'T_500', 'K_500')