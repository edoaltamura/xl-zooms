import sys
import numpy as np
from unyt import kpc
from matplotlib import pyplot as plt

sys.path.append("..")

from literature import Croston2008, Pratt2010
plt.style.use('../register/mnras.mplstyle')
lit = Croston2008()
lit.quick_display()
