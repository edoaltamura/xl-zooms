from radial_profiles_points import profile_3d_single_halo
from os.path import isfile
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

cwd = '/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/'

paths = {
    0.: (
        cwd + 'vr_partial_outputs/alpha0p0.properties',
        cwd + 'L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_alpha0p0/snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_2252.hdf5'
    ),
    0.5: (
        cwd + 'vr_partial_outputs/alpha0p5.properties',
        cwd + 'L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_alpha0p5/snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_2252.hdf5'
    ),
    0.7: (
        cwd + 'vr_partial_outputs/alpha0p7.properties',
        cwd + 'L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_alpha0p7/snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_2252.hdf5'
    ),
    0.9: (
        cwd + 'vr_partial_outputs/alpha0p9.properties',
        cwd + 'L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_alpha0p9/snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_2252.hdf5'
    ),
    1.: (
        cwd + 'vr_partial_outputs/alpha1p0.properties',
        cwd + 'L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_alpha1p0/snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_2252.hdf5'
    ),
}

fig, ax = plt.subplots()
name = "Set2"
cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap

fig, ax = plt.subplots()

for alpha_key in paths:
    print(alpha_key)
    vr, sw = paths[alpha_key]

    if ~isfile(vr) or ~isfile(sw):
        continue

    output = profile_3d_single_halo(
        path_to_snap=sw,
        path_to_catalogue=vr,
    )
    radial_distance, field_value, field_masses, field_label, M500, R500 = output

    ax.plot(radial_distance[::20], field_value[::20], marker=',', lw=0, linestyle="", c=cmap(alpha_key), alpha=0.1)

plt.show()

