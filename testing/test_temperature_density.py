import sys

sys.path.append("..")

from scaling_relations import TemperatureDensity
from test_files import cat, snap

gf = TemperatureDensity()
print(
    gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat,
        agn_time='after'
    )
)
