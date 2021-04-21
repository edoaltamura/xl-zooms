import sys

sys.path.append("..")

from scaling_relations import StarFractions
from test_files import cat, snap

gf = StarFractions()
print(
    gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat
    )
)

print(
gf.process_catalogue()
)