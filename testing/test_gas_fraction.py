import sys

sys.path.append("..")

from scaling_relations.gas_fraction import GasFraction

cwd = '/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/'
runname = 'L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1'
cat, snap = (
        cwd + 'vr_partial_outputs/alpha1p0.properties',
        cwd + runname + '_alpha1p0/snapshots/' + runname + '_SNnobirth_2252.hdf5'
    )

gf = GasFraction()
print(
    gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat
    )
)