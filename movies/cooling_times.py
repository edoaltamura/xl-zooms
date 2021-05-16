import sys

sys.path.append("..")

from scaling_relations import CoolingTimes
from register import parser

gf = CoolingTimes()

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
except Exception as e:
    print(f"Snap number {args.snapshot_number:04d} could not be processed.", e, sep='\n')