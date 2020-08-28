import numpy as np
import matplotlib.pyplot as plt
import os
import camb
from camb import model, initialpower, get_matter_power_interpolator

pars = camb.read_ini('planck_2018_eagleXL.ini')

# What redshifts do we want?
z_list = [2.0]
pars.set_matter_power(redshifts=z_list, kmax=15000.)

results = camb.get_results(pars)

print (np.array(results.get_sigma8()))

kh, zs, pk = results.get_matter_power_spectrum(minkh=1.e-4, maxkh=15000., npoints=2000)

np.savetxt('./raw_power_spectra/planck_2018_eagleXL_z2', np.c_[kh, pk[0]])

