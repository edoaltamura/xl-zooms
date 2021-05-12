from mpi4py import MPI

comm = MPI.COMM_WORLD
num_processes = comm.size
rank = comm.rank

import sys

sys.path.append("..")

from scaling_relations import CoolingTimes
from register import parser

parser.add_argument(
    '-s',
    '--snapshot-number',
    type=int,
    default=0
)

args = parser.parse_args()

dir = '/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_alpha1p0/'


# Data assignment can be done through independent operations
for snap_number in range(args.snapshot_number, 2523):
    if snap_number % num_processes == rank:
        print(f"Rank {rank:03d} processing snapshot {snap_number:03d}")

        s = dir + f"snapshots/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{snap_number:04d}.hdf5"
        c = dir + f"stf/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{snap_number:04d}/L0300N0564_VR18_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_{snap_number:04d}.properties"

        try:
            gf = CoolingTimes()
            gf.process_single_halo(
                path_to_snap=s,
                path_to_catalogue=c,
                agn_time=None,
                z_agn_start=18,
                z_agn_end=0
            )
        except:
            print(f"[!!!] Snap number {snap_number:04d} could not be processed.")