import sys

sys.path.append("..")

from scaling_relations import Entropies
from test_files import cat, snap

gf = Entropies()
single_halo_results = gf.process_single_halo(
        path_to_snap=snap,
        path_to_catalogue=cat
    )
for value in single_halo_results:
    print(value)

# print(
#     gf.process_catalogue()
# )
