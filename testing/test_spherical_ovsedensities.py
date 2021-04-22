import sys

sys.path.append("..")

from scaling_relations import SphericalOverdensities
from test_files import cat, snap

gf = SphericalOverdensities(density_contrast=1500.)
print(
    gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat
    )
)

# print(
#     gf.process_catalogue()
# )
