import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import swiftsimio as sw
import unyt

# EAGLE-XL data path
dataPath="/cosma7/data/dp004/jch/EAGLE-XL/DMONLY/Cosma7/L0300N0564/snapshots/"
# VR data path
vrPath=dataPath+"stf_swiftdm_3dfof_subhalo_0036/"
# Halo properties file
haloPropFile=vrPath+"stf_swiftdm_3dfof_subhalo_0036.VELOCIraptor.properties.0"
# Snapshot file
snapFile=dataPath+"EAGLE-XL_L0300N0564_DMONLY_0036.hdf5"

# Define mask
x=[134.688, 90.671, 71.962]
y=[169.921, 289.822, 69.291]
z=[289.233, 98.227, 240.338]
choice=0
rad=5.2

xCen=unyt.unyt_quantity(x[choice],unyt.Mpc)
yCen=unyt.unyt_quantity(y[choice],unyt.Mpc)
zCen=unyt.unyt_quantity(z[choice],unyt.Mpc)
size=unyt.unyt_quantity(1.*rad,unyt.Mpc)

mask=sw.mask(snapFile)
box=mask.metadata.boxsize
#region=[[0.8*b,0.9*b] for b in box]
region=[[xCen-size,xCen+size],[yCen-size,yCen+size],[zCen-size,zCen+size]]
mask.constrain_spatial(region)
# Load data using mask
data=sw.load(snapFile,mask=mask)
posDM=data.dark_matter.coordinates

plt.figure()
plt.plot(posDM[:,0]-xCen,posDM[:,1]-yCen,',')
plt.xlim([-size,size])
plt.ylim([-size,size])
plt.show()


