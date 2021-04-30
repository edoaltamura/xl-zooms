import sys

sys.path.append("..")

from scaling_relations import TemperatureDensity
from test_files import cat, snap

gf = TemperatureDensity()

# gf.process_single_halo(
#         path_to_snap=snap,
#         path_to_catalogue=cat,
#         agn_time=None,
#         z_agn_start=18,
#         z_agn_end=0
#     )
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

gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat,
        agn_time='before',
        z_agn_start=1,
        z_agn_end=0
    )

gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat,
        agn_time='before',
        z_agn_start=3,
        z_agn_end=1
    )

gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat,
        agn_time='before',
        z_agn_start=18,
        z_agn_end=3
    )
