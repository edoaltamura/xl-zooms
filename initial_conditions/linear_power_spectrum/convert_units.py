#!/bin/env python
#
# Convert CAMB output into units used in IC_GEN input files
#
# Input:  k, P(k)
# Output: log10(k), log10(k^3 P(k))
#

import sys
import scipy.interpolate
import numpy as np

if len(sys.argv) != 3:
    print("Usage: ./convert_units.py infile outfile")
infile = sys.argv[1]
outfile = sys.argv[2]

# Read input
data = np.loadtxt(infile)
k  = data[:,0]
pk = data[:,1]

# Convert
logk    = np.log10(k)
logk3pk = np.log10(k**3*pk)

# Extrapolate
interp = scipy.interpolate.interp1d(logk[:2], logk3pk[:2], fill_value="extrapolate")
logk_new = (-6.0, -5.0)
logk3pk_new = interp(logk_new)

# Combine with extrapolated points
logk    = np.concatenate((logk_new, logk))
logk3pk = np.concatenate((logk3pk_new, logk3pk))

# Output
f = open(outfile, "w")
for col1, col2 in zip(logk, logk3pk):
    f.write("%12.6e %12.6e\n" % (col1, col2))
f.close()
