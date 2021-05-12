import sys

sys.path.append("..")

from scaling_relations import CoolingTimes
from register import parser
from test_files import cat, snap

gf = CoolingTimes()

# gf.process_single_halo(
#     path_to_snap=snap,
#     path_to_catalogue=cat,
#     agn_time=None,
#     z_agn_start=18,
#     z_agn_end=0
# )
#
# gf.process_single_halo(
#         path_to_snap=snap,
#         path_to_catalogue=cat,
#         agn_time='before',
#         z_agn_start=18,
#         z_agn_end=0
#     )
#
# gf.process_single_halo(
#         path_to_snap=snap,
#         path_to_catalogue=cat,
#         agn_time='after',
#         z_agn_start=18,
#         z_agn_end=0
#     )

# gf.process_single_halo(
#         path_to_snap=snap,
#         path_to_catalogue=cat,
#         agn_time='before',
#         z_agn_start=1,
#         z_agn_end=0
#     )
#
# gf.process_single_halo(
#         path_to_snap=snap,
#         path_to_catalogue=cat,
#         agn_time='before',
#         z_agn_start=3,
#         z_agn_end=1
#     )
#
# gf.process_single_halo(
#         path_to_snap=snap,
#         path_to_catalogue=cat,
#         agn_time='before',
#         z_agn_start=18,
#         z_agn_end=3
#     )

# dir = '/cosma/home/dp004/dc-alta2/data6/xl-zooms/hydro/'

# snaps = [dir + i for i in [
#     "L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT7.5_Nheat1_SNnobirth/snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT7.5_Nheat1_SNnobirth_0036.hdf5",
#     "L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth/snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_0036.hdf5",
#     "L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8_Nheat1_SNnobirth/snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8_Nheat1_SNnobirth_0036.hdf5",
#     "L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT9.5_Nheat1_SNnobirth/snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT9.5_Nheat1_SNnobirth_2749.hdf5",
#     "L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT9_Nheat1_SNnobirth/snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT9_Nheat1_SNnobirth_0036.hdf5",
# ]
#          ]
#
# cats = [dir + i for i in [
#     "L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT7.5_Nheat1_SNnobirth/stf/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT7.5_Nheat1_SNnobirth_0036/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT7.5_Nheat1_SNnobirth_0036.properties",
#     "L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth/stf/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_0036/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_0036.properties",
#     "L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8_Nheat1_SNnobirth/stf/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8_Nheat1_SNnobirth_0036/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8_Nheat1_SNnobirth_0036.properties",
#     "L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT9.5_Nheat1_SNnobirth/stf/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT9.5_Nheat1_SNnobirth_2749/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT9.5_Nheat1_SNnobirth_2749.properties",
#     "L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT9_Nheat1_SNnobirth/stf/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT9_Nheat1_SNnobirth_0036/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT9_Nheat1_SNnobirth_0036.properties",
# ]
#          ]

parser.add_argument(
    '-s',
    '--snapshot-number',
    type=int,
    required=True
)

args = parser.parse_args()

dir = '/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_alpha1p0/'
s = dir + f"snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{args.snapshot_number:04d}.hdf5"
c = dir + f"stf/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{args.snapshot_number:04d}/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{args.snapshot_number:04d}.properties"

try:
    gf.process_single_halo(
        path_to_snap=s,
        path_to_catalogue=c,
        agn_time=None,
        z_agn_start=18,
        z_agn_end=0
    )
except:
    print(f"Snap number {args.snapshot_number:04d} could not be processed.")