import sys
import numpy as np
from unyt import kpc
from matplotlib import pyplot as plt

sys.path.append("..")

from literature import Croston2008, Pratt2010
plt.style.use('../register/mnras.mplstyle')
lit = Croston2008(disable_cosmo_conversion=True)
# lit.interpolate_r_r500(np.logspace(-2, 0, 1000))
lit.quick_display(quantity='Mgas')
