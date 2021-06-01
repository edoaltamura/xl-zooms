import sys

sys.path.append("..")

from scaling_relations import Entropies
from test_files import cat, snap

gf = Entropies()
single_halo_results = gf.process_single_halo(
    path_to_snap=snap,
    path_to_catalogue=cat
)
for value, key in zip(
        single_halo_results,
        ['k30kpc', 'k0p15r500', 'k2500', 'k1500', 'k1000', 'k500', 'k200']
):
    print(key, value)

# print(
#     gf.process_catalogue()
# )
