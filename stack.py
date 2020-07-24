# Compute expansion factor values for stacking boxes

import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import quad
from scipy import interpolate

# Relevant parameters - latest Planck parameters (last column of Table: 
# TT,TE,EE+lowE+lensing+BAO)
OmegaM = 0.3111  # Matter density parameter at z=0
h100 = 0.6766  # Hubble constant in 100 km/s/Mpc
BoxMpc = 300.  # Box-size in comoving Mpc
BoxFac = 1.  # Multiples of box-size to calculate output time

NumBoxes = 36

startingvalue = 0.  # initial distance in Mpc
print('Starting distance in Mpc =', startingvalue)

# Parameters for integral table
MinExpFacTab = 0.01
MaxExpFacTab = 1.0
NumTab = 10000  # Number of points in table for interpolation


# Integrand for calculating expansion factor values (flat universe only)
def expint(a):
    opz = 1. / a
    Eofz = np.sqrt(OmegaM * (opz ** 3) + 1. - OmegaM)
    result = 1. / (a * a * Eofz)
    return result


print('Cosmology: Omegam,h100 =', OmegaM, h100)

# Want to solve int_{a_N}^{1} da/[a^{2}*E(a)] = (N*h/3000.)*(box/Mpc)
RHSFac = h100 * BoxMpc / 3000.

print('RHS of integral equation for one box =', RHSFac)

# Calculate integral table for LHS with varying lower limit on a
StepSizeTab = (MaxExpFacTab - MinExpFacTab) / (NumTab - 1)
ExpFacTab = MinExpFacTab + np.arange(NumTab) * StepSizeTab
IntTab = np.zeros(NumTab)
for i in np.arange(NumTab):
    IntTab[i] = quad(expint, ExpFacTab[i], MaxExpFacTab)[0]
# Convert to equivalent number of slices
NumSlicesTab = IntTab / (RHSFac * BoxFac)

# Calculate expansion factor for integer number of slices (plus optional non-zero starting value)
# Start by creating interpolation function
interpfunc = interpolate.interp1d(NumSlicesTab, ExpFacTab, kind='cubic')

numslices = np.arange(NumBoxes) + 1
expfacslices = interpfunc(numslices + (startingvalue / (BoxFac * BoxMpc)))
zslices = (1. / expfacslices) - 1.

# Angle subtended by box at each redshift in degrees
thetamax = (180. / np.pi) * BoxMpc / ((numslices * BoxFac * BoxMpc) + startingvalue)

# Plot comoving distance versus redshift
# pl.figure()
# pl.plot((1./ExpFacTab)-1.,NumSlicesTab*BoxFac*BoxMpc)
# pl.plot(zslices,numslices*BoxFac*BoxMpc,'x',color='red')
# pl.xlabel('Redshift')
# pl.ylabel('Comoving Distance / Mpc')
# pl.xlim([0,20])
# pl.ylim([0,12000])
# pl.show()

for i in np.arange(numslices.size):
    print(format(numslices[i], '02d'), format(startingvalue + numslices[i] * BoxFac * BoxMpc, '5.0f'),
          format(zslices[i], '6.2f'), format(expfacslices[i], '.2f'))
