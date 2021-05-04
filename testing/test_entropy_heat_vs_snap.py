import sys

sys.path.append("..")

from scaling_relations import EntropyComparison
from test_files import cat, snap

gf = EntropyComparison()

gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat,
        agn_time='after',
        z_agn_start=1,
        z_agn_end=0
    )
